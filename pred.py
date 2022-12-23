import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import scipy as sp
import pandas as pd

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from utils import get_all_player_logs, refresh_oauth_file, fix_names_teams, get_team_ids, yahoo_to_nba_name

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

def return_all_taken_stats(league, sigma=10):
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
    tp = league.taken_players()

    # Build averaged stats for each player
    all_stats = {}
    for p in tp:
        name = yahoo_to_nba_name(p['name'])
        

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

def get_games_remaining(players, start_date, end_date):
    return

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
        name = yahoo_to_nba_name(p)
        for s in CORE_STATS:
            total_stats[s] += all_stats[name]*players[p] 

    return total_stats

if __name__ == "__main__":
    sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')
    stats = return_all_taken_stats(curLg)

    teamDF = get_team_ids(sc, curLg)

  
    week = curLg.current_week()
    startDate, endDate = curLg.week_date_range(week)
    dateDiff = endDate - startDate

    # get the date ranges with a timestamp of 11:59:59 PM; that way the day has ended
    # so all of the players in non-bench positions with a game will have played
    dateRanges = [datetime.datetime.combine(startDate + datetime.timedelta(days=d), datetime.time(23,59,59))
                for d in range(dateDiff.days + 2)]
    
    # get sunday from the week before as well to see if there were any add rights before the week started
    previousSunday = datetime.datetime.combine(startDate - datetime.timedelta(days=1), datetime.time(23,59,59))

    # get the roster for previous sunday
    teamDF[previousSunday] = teamDF['teamObject'].apply(lambda teamObject: pd.DataFrame(teamObject.roster(day = previousSunday)))
   

    # loop through the days and get the roster for each day
    for dIdx, currentDate in enumerate(dateRanges):
        
        # get the roster for the current date
        currentRoster = teamDF['teamObject'].apply(lambda teamObject: pd.DataFrame(teamObject.roster(day = currentDate)))