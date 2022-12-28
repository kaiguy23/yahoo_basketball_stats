import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import scipy as sp
import pandas as pd

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from utils import get_all_player_logs, refresh_oauth_file, fix_names_teams, get_team_ids, yahoo_to_nba_name, get_all_taken_players_extra

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

def project_stats_team(players, all_stats):
    """
    Projects the stats from unplayed games for the given week
    that the players were generated from

    Args:
        players (_type_): _description_
        all_stats (_type_): _description_
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
        projections[t] = predicted_total_stats(teams[t], all_stats)

    return projections




def predicted_total_stats(players, all_stats):
    """
    Returns the predicted total stats for a group of players

    Args:
        players (dict): {player_name: games to add together}
    """
    total_stats = {}
    for s in CORE_STATS:
        total_stats[s] = 0

    for p in players:
        name = players[p]['nba_name']
        for s in CORE_STATS:
            total_stats[s] += all_stats[name][s]*(players[p]["games_total"]-players[p]["games_played"])

    return total_stats

if __name__ == "__main__":
    sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')
    players = get_all_taken_players_extra(sc, curLg, curLg.current_week())
    stats = return_all_taken_stats(curLg, tp=players)
    proj = project_stats_team(players, stats)

    