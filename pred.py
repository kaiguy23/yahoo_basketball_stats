import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
from scipy.stats import skellam, poisson, norm
import pandas as pd
import copy
from pathlib import Path

from itertools import product

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from utils import get_all_player_logs, refresh_oauth_file, fix_names_teams, get_team_ids, yahoo_to_nba_name, get_all_taken_players_extra, extract_matchup_scores, TODAY

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

def return_all_taken_stats(league, sigma=10, tp=None, date=TODAY):
    """
    Returns average stats for all taken players (i.e. players on teams),
    weighting recent games more

    Args:
        league : class
            yahoo_fantasy_api.league.League
        sigma: number
            std of gaussian kernal (in number of games)
        tp: dict
            preload the taken players
        date: datetime object
            date to do the stat projects from (includes games only before this date)

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
            # Filter stats to only have games before the 
            # specified date
            stats = logs.get_group(name)
            game_dates = stats['GAME_DATE'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S").date())
            stats = stats[game_dates < TODAY]

            kernel = gkern_1sided(stats.shape[0],sigma)
            for s in CORE_STATS:
                to_add[s] = np.sum(stats[s].values*kernel)
        except:
            for s in CORE_STATS:
                to_add[s] = 0

        all_stats[name] = to_add

    return all_stats

def project_stats_team(players, all_stats, num_games=None,consider_status=True, 
                        count_IL = False, subtract_played=True, acutal_played=False):
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
        actual_played (bool): whether to consider the actual number of games played for backtesting

    RETURNS
        dict that maps team to dictionary of stat projections
    """


    # loop through players to build teams
    # each team is a dict mapping from 
    # fantasy manager to a dictionary of players
    # that are a subset of the players input
    teams = {}
    for p in players:
        # To record the actual number of games played
        # modify the games total/games played 
        # category and make copies for each team a player is on
        if acutal_played:
            counts = {}
            for i in range(len(players[p]['actual_played']['date'])):
                t = players[p]['actual_played']['manager'][i]
                if t in counts:
                    counts[t]+=1
                else:
                    counts[t]=1

            for t in counts:
                p_mod = copy.deepcopy(players[p])
                p_mod['games_total'] = counts[t]
                p_mod['games_played'] = counts[t]
                if t in teams:
                    teams[t][p] = p_mod
                else:
                    teams[t] = {p:p_mod}
        else:
            if players[p]['manager'] in teams:
                teams[players[p]['manager']][p] = players[p]
            else:
                teams[players[p]['manager']] = {p:players[p]}

    # loop through teams
    projections = {}
    for t in teams:
        if acutal_played:
            projections[t] = predicted_total_stats(teams[t], all_stats, consider_status=False,
                             count_IL=True, subtract_played=False)
        else:
            projections[t] = predicted_total_stats(teams[t], all_stats, consider_status=consider_status,
                             count_IL=count_IL, subtract_played=subtract_played)

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

def prob_victory(proj, p1, p2, matchup_df = None):
    """
    Returns the probability of victory in each category and overall
    between the two players

    Args:
        proj (dict): _description_
        p1 (str): key in proj, the name of player 1 to be compared
        p2 (str): key in proj, the name of player 2 to be compared
        matchup_df (pd dataframe): Dataframe that has the scores and matchups for the given week.
                                    Includes the current stats into the projections.
    
    Returns: np.array and dict
        (overall p1 victory prob, overall p2 victory prob, overall tie), {stat: (p1 victory, p2 victory tie)}
    """
    simple_stats = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV"]

    percent_stats = {"FG%": ("FGA", "FGM"), "FT%": ("FTA", "FTM")}

    stat_victory = {}

    if not matchup_df is None:
        grouped = matchup_df.groupby("manager")

    # Go through simple counting stats
    for stat in simple_stats:
        if matchup_df is None:
            current_score = (0,0)
        else:
            current_score = (grouped.get_group(p1)[stat].iloc[0], grouped.get_group(p2)[stat].iloc[0])
        stat_victory[stat] = skellam_prob(proj[p1][stat], proj[p2][stat], current_score=current_score)
        if stat == "TOV":
            w1 = stat_victory[stat][1]
            w2 = stat_victory[stat][0]
            stat_victory[stat] = (w1, w2, stat_victory[stat][2])

    # Go through percentage stats
    percent_std = {}
    for stat in percent_stats:
        attempts = (proj[p1][percent_stats[stat][0]], proj[p2][percent_stats[stat][0]])
        made = (proj[p1][percent_stats[stat][1]], proj[p2][percent_stats[stat][1]])

        if matchup_df is None:
            current_score = ((0,0),(0,0))
        else:
            current_score = ((grouped.get_group(p1)[percent_stats[stat][0]].iloc[0], grouped.get_group(p2)[percent_stats[stat][0]].iloc[0]),
                            (grouped.get_group(p1)[percent_stats[stat][1]].iloc[0], grouped.get_group(p2)[percent_stats[stat][1]].iloc[0]))
        stat_victory[stat], moment1, moment2 = ratio_prob(attempts, made, current_score=current_score)
        percent_std[stat] = {}
        percent_std[stat][p1] = moment1
        percent_std[stat][p2] = moment2



    probs = np.zeros(3)

    # Loop through all 19,683 possible stat winning combinations/ties
    # iterate over all lists of 9 zeros (p1 victory) and ones (p2 victory), and twos (ties)
    for combo in product(np.arange(3), repeat=9):
        p = 1
        for i, stat in enumerate(stat_victory):
            p*=stat_victory[stat][combo[i]]
        players, wins = np.unique(combo, return_counts=True)
        # One player gets no wins
        if 1 not in players:
            probs[0]+=p
        elif 0 not in players:
            probs[1]+=p
        # Tie
        elif wins[0] == wins[1]:
            probs[2]+=p
        # Normal matchups
        else:
            if wins[0] > wins[1]:
                probs[0]+=p
            else:
                probs[1]+=p 
    
    # Normalize to smooth out numerical relics
    probs/=np.sum(probs)
    
    return probs, stat_victory, percent_std

def ratio_prob(attempts, made, samples=10000, current_score = ((0,0), (0,0))):
    """
    Randomly samples attempts and made as two independent Poisson distributions
    to get a probability of winning the percentages


    Args:
        attempts (tuple): number of attempts (p1, p2)
        made (tuple): number made (p1, p2)
        samples (int, optional): number of samples for estimating distribution. Defaults to 10000.
        current_score (tuple): current score ((p1 attempts, p2 attempts), (p1 made, p2 made))

    Returns:
        three tuples, (prob p1 victory, prob p2 victory, prob tie), (p1 mean, p1 std), (p2 mean, p2 std)
    """

    # Attempts as Poisson process
    a1 = np.random.poisson(attempts[0], samples) + current_score[0][0]
    a2 = np.random.poisson(attempts[1], samples) + current_score[0][1]
    a1[a1 == 0] = 1
    a2[a2 == 0] = 1

    # Made as Poisson process
    m1 = np.random.poisson(made[0], samples) + current_score[1][0]
    m2 = np.random.poisson(made[1], samples) + current_score[1][1]

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

def ratio_range(attempts, made, samples=10000, current_score=(0,0)):

    # Attempts as Poisson process
    a = np.random.poisson(attempts, samples) + current_score[0]
    m = np.random.poisson(made, samples) + current_score[1]
   
    r = m/a
    r[r > 1] = 1
    
    return np.mean(r), np.std(r)



def skellam_prob(mu1, mu2, current_score=(0,0)):
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

    # Shift x to represent the current score
    x += current_score[0]-current_score[1]

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
    

def ideal_matrix(proj, num_games = 3, savename="ideal.png", matchup_df = None, week=""):
    """
    Generates a matrix of predicted outcomes if every player played every other player

    Args:
        proj (dict): dictionary of manager name to projected stat values
        num_games (int, optional): _description_. Defaults to 3.
        savename (str, optional): _description_. Defaults to "ideal.png".
        matchup_df (pd dataframe, optional): 

    Returns:
        _type_: _description_
    """

    if matchup_df is None:
        managers = list(proj.keys())
    else:
        managers = matchup_df['manager'].values

    probMat = np.zeros((len(managers), len(managers)))
    for i1, m1 in enumerate(managers):
        for i2, m2 in enumerate(managers):
            if i2 > i1:
                p, s, m = prob_victory(proj, m1, m2, matchup_df=matchup_df)
                probMat[i1, i2] = p[0]
                probMat[i2, i1] = p[1]
            elif i2 == i1:
                probMat[i1, i2] = np.nan


    

    # create labels for the axes
    if matchup_df is None:
        yAxisLabels = managers
    else:
        yAxisLabels = matchup_df[['manager', 'teamName']].apply(lambda x: x[0] + '\n' + x[1],axis=1)

    xAxisLabels = managers

    # do plotting
    sn.set(font_scale=1.2)
    f, ax = plt.subplots(figsize=(20,10))
    ax = sn.heatmap(probMat, annot=np.round(probMat*100)/100, fmt='', xticklabels = xAxisLabels,
            yticklabels = yAxisLabels, cmap='RdYlGn',cbar=False)

    # highlight actual matchup
    if not matchup_df is None:
        # add in patches to mark who actually played who in that week
        # get number of unique matchups:
        for m in matchup_df['matchupNumber'].unique():
            i,j = matchup_df[matchup_df['matchupNumber']==m].index
            ax.add_patch(Rectangle((i,j), 1, 1, fill=False, edgecolor='blue', lw=3))
            ax.add_patch(Rectangle((j,i), 1, 1, fill=False, edgecolor='blue', lw=3))

    if num_games is None:
        if week == "":
            f.suptitle(f"NBA Fantasy Predicted Results", fontsize = 30)
        else:
            f.suptitle(f"NBA Fantasy Predicted Results (Week {week})", fontsize = 30)
    else:
        f.suptitle(f"NBA Fantasy Ideal Matchups (All Players Play {num_games} Games, Ignore Injuries, Don't Count Players on IL)", fontsize = 30)

    if savename != "":
        plt.savefig(savename)
        plt.close(f)

    return probMat

def plot_matchup_summary(proj, p1, p2, matchup_df = None, savename=None):
    """
    Plots a summary of the specified matchup, showing the 90% confidence interval
    for each stat, showing the probability that either player will 
    win the category.


    Args:
        proj (dict): dictionary that maps player to projected stats
        p1 (str): name of the first player
        p2 (str): name of the second player
        matchup_df (pandas df, optional): matchup df with the current score for midweek analysis. Defaults to None.
    """

    p,s,m = prob_victory(proj, p1, p2, matchup_df=matchup_df)
    
    f, ax = plt.subplots(nrows=3,ncols=3, figsize=(20,14))

    vic = [np.round(x*100,2) for x in p]
    f.suptitle(f"Probability of Victory - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%", fontsize=26)

    # epsilon = 0.0001
    epsilon = 0.05

    if not matchup_df is None:
        grouped = matchup_df.groupby("manager")

    
   
    for ip, p in enumerate((p1,p2)):
        for i, stat in enumerate(s):
            if "%" in stat:
                mu = m[stat][p][0]
                sigma = m[stat][p][1]
                x = (norm.ppf(epsilon, loc=mu, scale=sigma),
                    norm.ppf(1-epsilon,loc=mu, scale=sigma))
                # prob = norm.pmf(x, loc=mu, scale=sigma)
            else:
                mu = proj[p][stat]
                x = np.arange(poisson.ppf(epsilon, mu),
                    poisson.ppf(1-epsilon, mu)+1)
                # prob = poisson.pmf(x, mu)
            row = int(i/3)
            col = i-int(i/3)*3
            a = ax[row,col]

            if ip == 0:
                vic = np.array([np.round(x*100,2) for x in s[stat]])
                a.set_title(f"{stat} - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%")
            # Add current stats
            exp_val = mu
            if "%" not in stat and not matchup_df is None:
                exp_val+=grouped.get_group(p)[stat].iloc[0]
            a.errorbar(exp_val, (1-ip)*0.01, xerr=np.array((mu-x[0], x[-1]-mu)).reshape(2,1), label=p, capsize = 4, fmt = 'o', alpha=0.75)
            a.set_ylim([-0.005,0.015])
            a.set_yticks([])
            if ip == 1:
                a.legend()
    
    if savename is None:
        savename = f"{p1}_vs_{p2}.png"
    plt.savefig(savename)

    return


def past_preds(sc, gm, curLg, week, savename=None):
    """
    Does the predictions as if they were at the start of the last week

    Args:
        week (int): week to test

    returns:
        dict proj for the week
        matchup_df showing results for the week
    """

    # sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')

    d0 = curLg.week_date_range(week)[0]

    # print("Predictions for week", week, "from dates:", curLg.week_date_range(week))

    
    players = get_all_taken_players_extra(sc, curLg, week, actual_played=True, include_today=True)
    matchup_df = extract_matchup_scores(curLg, week, nba_cols=True)

    # Zero out matchup_df 
    matchup_df_blank = matchup_df.copy()
    for stat in CORE_STATS:
        if stat in matchup_df_blank.columns:
            matchup_df_blank[stat] = 0


    stats = return_all_taken_stats(curLg, tp=players, date=d0)
    
    proj = project_stats_team(players, stats, acutal_played=True)

    if not savename is None:
        probMat = ideal_matrix(proj, num_games=None, 
                    savename=savename, matchup_df=matchup_df_blank, week=week)


    return proj, matchup_df



def run_predictions(sc, gm, curLg, week, folder, midweek=False):

    players = get_all_taken_players_extra(sc, curLg, week, include_today=False)

    matchup_df = extract_matchup_scores(curLg, week, nba_cols=True)
    stats = return_all_taken_stats(curLg, tp=players)

    # Reset the stats
    # Zero out matchup_df 
    matchup_df_blank = matchup_df.copy()
    for stat in CORE_STATS:
        if stat in matchup_df_blank.columns:
            matchup_df_blank[stat] = 0
    
    proj = project_stats_team(players, stats, subtract_played=False)
    grouped = matchup_df.groupby("matchupNumber")
    for i in grouped.indices:
        matchup = grouped.get_group(i)
        p1 = matchup['manager'].iloc[0]
        p2 = matchup['manager'].iloc[1]
        savename = str(Path(folder,f"{p1}_vs_{p2}.png"))
        plot_matchup_summary(proj,p1, p2, matchup_df=matchup_df_blank, savename=savename)
        # plot_matchup_summary(proj,p1, p2, savename=savename)
        
    probMat = ideal_matrix(proj, num_games=None, savename=Path(folder,"pred_mat.png"), matchup_df=matchup_df_blank, week=week)



if __name__ == "__main__":
    sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')

    week = curLg.current_week()

    proj, matchup_df = past_preds( sc, gm, curLg, week-1, savename="past_preds.png")

    # run_predictions(sc, gm, curLg, week, "predictions")
    assert(False)
    
    ## TODO: FIX PLAYED TODAY OR NOT
    players = get_all_taken_players_extra(sc, curLg, week, include_today=False)


    matchup_df = extract_matchup_scores(curLg, week, nba_cols=True)
    stats = return_all_taken_stats(curLg, tp=players)
    
    proj = project_stats_team(players, stats, subtract_played=True)
    # # p, s, m  = prob_victory(proj, "Eli", "Chi Yen")    
    # # plot_matchup_summary(proj, "Eli", "Chi Yen", matchup_df=matchup_df)
    plot_matchup_summary(proj, "Kayla", "Gary", matchup_df=matchup_df)
    # plot_matchup_summary(proj, "Fabio", "Yi Sheng", matchup_df=matchup_df)

    # # # proj = project_stats_team(players, stats, num_games=4,count_IL=False, consider_status=False)
    # # # probMat = ideal_matrix(proj, num_games=None, savename="actual_last_week.png")
    probMat = ideal_matrix(proj, num_games=None, savename="actual.png", matchup_df=matchup_df, week=week)



    