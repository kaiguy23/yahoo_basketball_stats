import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
from scipy.stats import skellam, poisson, norm
import pandas as pd
import copy
from pathlib import Path

import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from itertools import product

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

import seaborn as sn

from db_interface import dbInterface
import utils



CORE_STATS = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
       'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK','PTS','NBA_FANTASY_PTS']


GKERN_SIG = 10


def gkern_1sided(size: int, sig: float = GKERN_SIG) -> np.array:
    """
    Creates a one sided gaussian kernel of length size,
    with peak at the 0 index, and std of sigma.

    Kernel is normalized to sum to 1.

    Args:
        size (int): length of kernel to produce
        sig (float, optional): Sigma (std) for Gaussian. Defaults to GKERN_SIG.

    Returns:
        np.array[float]: Array of length size with value 1 at index 0,
                         and a Gaussian falloff with std sig.
    """
    x = np.arange(size)
    gauss = np.exp(-0.5 * (x**2) / (sig**2))
    return gauss/np.sum(gauss)


def skellam_prob(mu1: float, mu2: float,
                 current_score: tuple[int] = (0,0)) -> tuple[float]:
    """
    Calculates the probability of mu1 winning
    or mu2 winning or it being a tie

    Args:
        mu1 (float): mean of first Poisson distribution
        mu2 (float): mean of second Poisson distribution

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


def ratio_prob(attempts: tuple[int], made: tuple[int],
               samples:int = 10000,
               current_score: tuple[tuple[int]] = ((0,0), (0,0))) -> (tuple[float],
                                                                      tuple[float],
                                                                      tuple[float]):
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


def proj_player_stats(db: dbInterface, name: str,
                      date: str = utils.TODAY_STR,
                      kern_sig: float = GKERN_SIG) -> dict[float]:
    """
    Projects player stats by taking a weighted average by a Gaussian
    kernel of std kern_sig of games played up to (but not including)
    the specified date.

    Args:
        db (dbInterface): db interface object
        name (str): name of player
        date (str, optional): Date in form YYYY-MM-DD.
                              Defaults to utils.TODAY_STR.
        kern_sig (float, optional): STD of Gaussian kernel. Defaults to GKERN_SIG.

    Returns:
        dict[float]: Maps statistic to projected value
    """

    # Get player stats up to the specified day
    stats = db.player_stats(name)
    stats = stats[stats["GAME_DATE"] < date]

    proj = {}
    # Reverse it because most recent games are last
    kern = gkern_1sided(stats.shape[0], kern_sig)[::-1]
    for cat in utils.STATS_COLS:
        proj[cat] = np.sum(kern*stats[cat].values)
    
    return proj
        

def proj_all_players(db: dbInterface, date: str = utils.TODAY_STR,
                     rostered: bool = True,
                     kern_sig: float = GKERN_SIG) -> pd.DataFrame:
    """
    Returns a dataframe with projected stats for all rostered
    or optionally all nba players on the morning of the specified 
    date (i.e., before any games have been played)

    Args:
        db (dbInterface): db Interface object
        date (str, optional): Date in form YYYY-MM-DD.
                              Defaults to utils.TODAY_STR.
        rostered (bool, optional): True -> return stats for only players rostered in fantasy
                                   False -> return stats for all players rostered in the NBA
        kern_sig (float, optional): STD of Gaussian kernal. Defaults to GKERN_SIG.

    Returns:
        pd.DataFrame: Dataframe with player names, projected stats,
                      and their fantasy status.
    """
    rosters = db.fantasy_rosters(date)
    df = []
    for i, row in rosters.iterrows():
        entry = {}
        entry["name"] = row["name"]
        entry.update(proj_player_stats(db, row["name"], date, kern_sig))
        to_copy = ["selected_position", "status", "manager", "teamName", "teamID"]
        for col in to_copy:
            entry[col] = row[col]
        df.append(entry)
            
    if not rostered:
        for name in db.fantasy_free_agents(date):
            entry = {}
            entry["name"] = row["name"]
            entry.update(proj_player_stats(db, row["name"], date, kern_sig))
            to_copy = ["selected_position", "status", "manager", "teamName", "teamID"]
            for col in to_copy:
                entry[col] = ""
            df.append(entry)
    
    return pd.DataFrame(df)


    




def prob_victory(proj: dict[str, tuple[float]],
                 current_scores: dict[str, tuple[float]] = {"PTS": (0, 0), "FG3M": (0, 0), 
                                                            "REB": (0, 0), "AST": (0, 0),
                                                            "ST": (0, 0), "BLK": (0, 0),
                                                            "TO": (0, 0), "FGA": (0, 0),
                                                            "FGM": (0, 0), "FTA": (0, 0),
                                                            "FTM": (0, 0)}) -> (np.array,
                                                                                dict[str, tuple[float]]):
    """
    Returns the probability of victory in each category and overall
    between the two players

    Args:
        proj (dict): maps stat category to a tuple of player 1 and player 2 projected values
        current_scores (dict): maps stat category to a tuple of player 1 and player 2 values

    Returns: np.array and dict
        (overall p1 victory prob, overall p2 victory prob, overall tie),
        {stat: (p1 victory, p2 victory tie)}
    """

    simple_stats = ["PTS", "3PTM", "REB", "AST", "ST", "BLK", "TO"]
    percent_stats = {"FG%": ("FGA", "FGM"), "FT%": ("FTA", "FTM")}

    stat_victory = {}

    if not matchup_df is None:
        grouped = matchup_df.groupby("manager")

    # Go through simple counting stats
    for stat in simple_stats:
        if curr is None:
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


    # Loop through all 19,683 possible stat winning combinations/ties
    # iterate over all lists of 9 zeros (p1 victory) and ones (p2 victory), and twos (ties)
    probs = np.zeros(3)
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

    
    db = dbInterface("past_season_dbs/yahoo_fantasy_2022_23.sqlite")
    date = "2023-03-23"
    proj = proj_all_players(db, date)


    # sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')

    # week = curLg.current_week()

    # # proj, matchup_df = past_preds( sc, gm, curLg, week-1, savename="past_preds.png")

    # # run_predictions(sc, gm, curLg, week, "predictions")
    
    # ## TODO: FIX PLAYED TODAY OR NOT
    # players = get_all_taken_players_extra(sc, curLg, week, include_today=False)


    # matchup_df = extract_matchup_scores(curLg, week, nba_cols=True)
    # # matchup_df = None
    # stats = return_all_taken_stats(curLg, tp=players)
    
    # proj = project_stats_team(players, stats, subtract_played=True)

    # # # p, s, m  = prob_victory(proj, "Eli", "Chi Yen")    
    # plot_matchup_summary(proj, "Eli", "Jack", matchup_df=matchup_df)
    # plot_matchup_summary(proj, "David", "Gary", matchup_df=matchup_df)
    # # plot_matchup_summary(proj, "Fabio", "Yi Sheng", matchup_df=matchup_df)
    # # plot_matchup_summary(proj, "Fabio", "Yi Sheng", matchup_df=matchup_df)

    # # # # proj = project_stats_team(players, stats, num_games=4,count_IL=False, consider_status=False)
    # # # # probMat = ideal_matrix(proj, num_games=None, savename="actual_last_week.png")
    # probMat = ideal_matrix(proj, num_games=None, savename=f"preds_{TODAY}.png", matchup_df=matchup_df, week=week)



    