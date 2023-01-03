import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
from scipy.stats import skellam
import pandas as pd

from itertools import product

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from utils import get_all_player_logs, refresh_oauth_file, fix_names_teams, get_team_ids, yahoo_to_nba_name, get_all_taken_players_extra, extract_matchup_scores

import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sn

CORE_STATS = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
       'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK','PTS','NBA_FANTASY_PTS']

def gkern_1sided(l, sig):
    """
    creates a one sided gaussian kernel with peak at the 0 index,
    and std of sigma
    """
    x = np.arange(l)
    gauss = np.exp(-0.5 * (x**2) / (sig**2))
    return gauss/np.sum(gauss)

def return_all_taken_stats(league, sigma=10, tp=None):
    """
    Returns average stats for all taken players (i.e. players on teams),
    weighting recent games more

    Args:
        league : class
            yahoo_fantasy_api.league.League
        sigma: number
            std of gaussian kernal (in number of games)

    RETURNS:
        dictionary that maps player names to stats
    """
    # Get all the stats for the whole season
    logs = get_all_player_logs()

    # Get all the players currently on teams
    if tp is None:
        tp = league.taken_players()
    

    # Build averaged stats for each player
    all_stats = {}
    for p in tp:

        # Save lookup time if we already got the nba name
        if 'nba_name' in tp[p].keys():
            name = tp[p]['nba_name']
        else:
            name = yahoo_to_nba_name(tp[p]['name'])
        

        to_add = {}

        # Catch cases where people have players rostered
        # that haven't played yet this season
        try:
            stats = logs.get_group(name)
            kernel = gkern_1sided(stats.shape[0],sigma)
            for s in CORE_STATS:
                to_add[s] = np.sum(stats[s].values*kernel)
        except:
            for s in CORE_STATS:
                to_add[s] = 0

        all_stats[name] = to_add

    return all_stats

def project_stats_team(players, all_stats, num_games=None,consider_status=True, count_IL = False, subtract_played=False):
    """
    Projects the stats from unplayed games for the given week
    that the players were generated from

    Args:
        players (dict): maps players to information, output from utils/get all taken players extra
        all_stats (dict): maps players to averaged stat outputs 
        num_games (int): hard code the number of games per player, for an unbiased comparison
        consider_status (bool): whether to take player status into account or not (i.e. injured or not)
        count_IL (bool): whether to count players on the IL or not
        subtract_played (bool): whether to subtract games played or not

    RETURNS
        dict that maps team to dictionary of stat projections
    """


    # loop through players to build teams
    # each team is a dict mapping from 
    # fantasy manager to a dictionary of players
    # that are a subset of the players input
    teams = {}
    for p in players:
        if players[p]['manager'] in teams:
            teams[players[p]['manager']][p] = players[p]
        else:
            teams[players[p]['manager']] = {p:players[p]}

    # loop through teams
    projections = {}
    for t in teams:
        projections[t] = predicted_total_stats(teams[t], all_stats, consider_status=consider_status, count_IL=count_IL, subtract_played=subtract_played)

    return projections

def predicted_total_stats(players, all_stats, num_games=None, consider_status=True, count_IL = False, subtract_played=False):
    """
    Returns the predicted total stats for a group of players

    Args:
        players (dict): {player_name: games to add together}
        all_stats (dict): maps players to averaged stat outputs 
        num_games (int): hard code the number of games per player, for an unbiased comparison
        consider_status (bool): whether to take player status into account or not (i.e. injured or not)
        count_IL (bool): whether to count players on the IL or not
        subtract_played (bool): whether to subtract games played or not

    RETURNS:
        dict that maps stat category to sum of total stats
    """
    total_stats = {}
    for s in CORE_STATS:
        total_stats[s] = 0

    for p in players:
        # Player on IL
        if not(count_IL) and 'IL' in players[p]['selected_position']:
            # print("IL:", p)
            continue
        name = players[p]['nba_name']

        # Multiplier for number of games played
        if num_games is None:
            if subtract_played:
                mult = players[p]["games_total"]-players[p]["games_played"]
            else:
                mult = players[p]["games_total"]
            if consider_status:
                status = players[p]['status']
                if status == 'INJ':
                    # print("INJ:", p)
                    mult = 0
                elif status == 'O':
                    # print("O:", p)
                    if mult > 0:
                        mult += -1
        else:
            mult = num_games
        for s in CORE_STATS:
            
            

            total_stats[s] += all_stats[name][s]*mult

    return total_stats

def prob_victory(proj, p1, p2):
    """
    Returns the probability of victory in each category and overall
    between the two players

    Args:
        proj (dict): _description_
        p1 (str): key in proj, the name of player 1 to be compared
        p2 (str): key in proj, the name of player 2 to be compared
    
    Returns: np.array and dict
        (overall p1 victory prob, overall p2 victory prob, overall tie), {stat: (p1 victory, p2 victory tie)}
    """
    simple_stats = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV"]

    percent_stats = {"FG%": ("FGA", "FGM"), "FT%": ("FTA", "FTM")}

    stat_victory = {}

    # Go through simple counting stats
    for stat in simple_stats:
        stat_victory[stat] = skellam_prob(proj[p1][stat], proj[p2][stat])
        if stat == "TOV":
            w1 = stat_victory[stat][1]
            w2 = stat_victory[stat][0]
            stat_victory[stat] = (w1, w2, stat_victory[stat][2])

    # Go through percentage stats
    percent_std = {}
    for stat in percent_stats:
        attempts = (proj[p1][percent_stats[stat][0]], proj[p2][percent_stats[stat][0]])
        made = (proj[p1][percent_stats[stat][1]], proj[p2][percent_stats[stat][1]])
        stat_victory[stat], moment1, moment2 = ratio_prob(attempts, made)
        percent_std[stat] = {}
        percent_std[stat][p1] = moment1
        percent_std[stat][p2] = moment2



    probs = np.zeros(3)

    # Loop through all 512 possible stat winning combinations, prob tie is 1 - pwin1 - pwin2
    # iterate over all lists of 9 zeros (p1 victory) and ones (p2 victory)
    for combo in product(np.arange(2), repeat=9):
        p = 1
        for i, stat in enumerate(stat_victory):
            p*=stat_victory[stat][combo[i]]
        if sum(combo) < 5:
            probs[0]+=p
        else:
            probs[1]+=p
    
    probs[2] = 1 - np.sum(probs)

    return probs, stat_victory, percent_std

def ratio_prob(attempts, made, samples=10000):
    """
    Randomly samples attempts and made as two independent Poisson distributions
    to get a probability of winning the percentages


    Args:
        attempts (_type_): _description_
        made (_type_): _description_
        samples (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """

    # Attempts as Poisson process
    a1 = np.random.poisson(attempts[0], samples)
    a2 = np.random.poisson(attempts[1], samples)
    a1[a1 == 0] = 1
    a2[a2 == 0] = 1

    # Made as Poisson process
    m1 = np.random.poisson(made[0], samples)
    m2 = np.random.poisson(made[1], samples)

    # Ratios i.e. percentage
    r1 = m1/a1
    r1[r1 > 1] = 1
    
    r2 = m2/a2
    r2[r2 > 1] = 1
    
    comp = r1 - r2


    w1 = np.sum(comp > 0)/samples
    tie = np.sum(comp == 0)/samples
    w2 = np.sum(comp < 0)/samples
    
    return (w1, w2, tie), (np.mean(r1), np.std(r1)), (np.mean(r2), np.std(r2))



def skellam_prob(mu1, mu2):
    """
    Calculates the probability of mu1 winning
    or mu2 winning or it being a tie

    Args:
        mu1 (number): mean of first Poisson distribution
        mu2 (number): mean of second Poisson distribution

    returns:
        tuple (probability dist 1 is higher, probability dist 2 is higher, prob of equal values)
    """
    # Get the x range to investigate
    epsilon = 0.0001
    x = np.arange(skellam.ppf(epsilon, mu1, mu2),
                    skellam.ppf(1-epsilon, mu1, mu2)+1)
    prob = skellam.pmf(x, mu1, mu2)

    # dist 2 > dist 1, add 0.01 because of the percentile bounds
    w2 = np.sum(prob[x < 0]) + epsilon

    # tie
    tie = np.sum(prob[x==0])

    # dist 1 > dist 2, add 0.01 because of the percentile bounds
    w1 = np.sum(prob[x > 0]) + epsilon

    # super high prob correction
    if 1 - w1 < epsilon:
        w2 = epsilon/2
        tie = epsilon/2
    if 1 - w2 < epsilon:
        w1 = epsilon/2
        tie = epsilon/2

    return (w1, w2, tie)
    

def ideal_matrix(proj, num_games = 3, savename="ideal.png"):
    """
    Generates a matrix of predicted outcomes if every player played every other player

    Args:
        proj (dict): dictionary of manager name to projected stat values
        num_games (int, optional): _description_. Defaults to 3.
        savename (str, optional): _description_. Defaults to "ideal.png".

    Returns:
        _type_: _description_
    """

    managers = list(proj.keys())
    probMat = np.zeros((len(managers), len(managers)))
    for i1, m1 in enumerate(managers):
        for i2, m2 in enumerate(managers):
            if i2 > i1:
                p, s, m = prob_victory(proj, m1, m2)
                probMat[i1, i2] = p[0]
                probMat[i2, i1] = p[1]
            elif i2 == i1:
                probMat[i1, i2] = np.nan


    

    # create labels for the axes
    yAxisLabels = managers
    xAxisLabels = managers

    # do plotting
    sn.set(font_scale=1.2)
    f, ax = plt.subplots(figsize=(20,10))
    ax = sn.heatmap(probMat, annot=np.round(probMat*100)/100, fmt='', xticklabels = xAxisLabels,
            yticklabels = yAxisLabels, cmap='RdYlGn',cbar=False)

    if num_games is None:
        f.suptitle(f"NBA Fantasy Predicted Results", fontsize = 30)
    else:
        f.suptitle(f"NBA Fantasy Ideal Matchups (All Players Play {num_games} Games, Ignore Injuries, Don't Count Players on IL)", fontsize = 30)

    if savename != "":
        plt.savefig(savename)
        plt.close(f)

    return probMat



if __name__ == "__main__":
    sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')
    players = get_all_taken_players_extra(sc, curLg, curLg.current_week())
    stats = return_all_taken_stats(curLg, tp=players)
    
    proj = project_stats_team(players, stats)
    # p, s, m  = prob_victory(proj, "Eli", "Chi Yen")    

    # proj = project_stats_team(players, stats, num_games=4,count_IL=False, consider_status=False)
    probMat = ideal_matrix(proj, num_games=None, savename="actual.png")

    