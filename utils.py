import os
import json
import yaml 
import math
import numpy as np
import pandas as pd
import datetime
from typing import Union

from yahoo_oauth import OAuth2

import yahoo_fantasy_api as yfa


from nba_api.stats.endpoints import playergamelogs, playercareerstats
from nba_api.stats.static.players import find_players_by_first_name, find_players_by_full_name, find_players_by_last_name
from nba_api.stats.endpoints import scoreboard

# Get the default season
TODAY = datetime.date.today()
if TODAY.month >= 10:
    DEFAULT_SEASON = f"{TODAY.year}_{str(TODAY.year+1)[-2:]}"
else:
    DEFAULT_SEASON = f"{TODAY.year-1}_{str(TODAY.year)[-2:]}"

DATE_SCHEMA = "%Y-%m-%d"
TODAY_STR = TODAY.strftime(DATE_SCHEMA)

# Map of NBA API stat column labels to change
NBA_TO_YAHOO_STATS = {"TOV": "TO",
                      "FG3M": "3PTM",
                      "FG3A": "3PTA",
                      "FG3_PCT": "3PT%",
                      "FG_PCT": "FG%",
                      "STL": "ST"}

STATS_COLS = ["3PTM", "PTS",
              "REB", "AST", "ST", "BLK",
              "TO", "FGM", "FGA", "FTM",
              "FTA"]

PERC_STATS = ["FG%", "FT%"]

ACTIVE_POS = ['C', 'PF', 'PG', 'SF',
              'F', 'SG', 'Util', 'G']

NBA_TEAMS = ['ATL','BKN','BOS','CHA','CHI','CLE',
             'DAL','DEN','DET','GSW','HOU','IND',
             'LAC','LAL','MEM','MIA','MIL','MIN',
             'NOP','NYK','OKC','ORL','PHI','PHX',
             'POR','SAC','SAS','TOR','UTA','WAS']
NBA_TEAMS_SET = set(NBA_TEAMS)

ROSTER_SPOTS = {"PG": 1, "SG": 1, "G": 1, "SF": 1,
                "PF": 1, "F": 1, "C": 2, "Util": 2}


SPECIAL_NAMES = {}
def yahoo_to_nba_name(name: str, hardcoded: dict = SPECIAL_NAMES) -> str:
    """
    Converts a yahoo API name to NBA api name. By in large they have the
    same name, but some players with abbreviations, like OG Anunoby
    (or O.G. Anunoby as the NBA api believes) have some inconsistencies.

    Args:
        name (str): yahoo api player name
        hardcoded (dict, optional): Dictionary to hardcode names not found
                                    automatically.

    Raises:
        ValueError: If player is not found

    Returns:
        str: NBA api name
    """
    # hardcoded 
    if name in hardcoded:
        return hardcoded[name]
    # Everything matches
    try:
        player = find_players_by_full_name(name)
        return player[0]['full_name']

    # Try matching first and last names and seeing if there's only one result
    # spit it out to be manually hard coded if not
    except:
        try:
            player = find_players_by_last_name(name.split(" ")[-1])
            if len(player) == 1:
                return player[0]['full_name']                
            else:
                player = find_players_by_first_name(name.split(" ")[0])
                if len(player) == 1:
                    return player[0]['full_name']
        except:
            try:
                player = find_players_by_first_name(name.split(" ")[0])
                if len(player) == 1:
                    return player[0]['full_name']
            except Exception as e:
                print(e)
                raise ValueError(f"Player {name} not found")


# In practice a little bit ~10 % faster for our use case,
# but probably not worth the extra complexity
def find_closest_date_fast(d: Union[str, datetime.datetime],
                      dates: list[str]) -> int:
    """
    NOTE: dates must be sorted already, with most
    recent dates at the end of the list

    Finds the index of the closest date in
    dates to the date d. Assumes dates is sorted
    already with most recent dates at the end of the 
    list.
    
    If d/dates
    are strings, assumes they are in the default
    date format.

    In the case of a tie, the winner is not deterministically
    chosen.

    Args:
        d (str or datetime): date to find the closest entry to
        dates (list of str or datetime): list of dates to compare to
    
    Returns:
        int: index of the closest date in dates to the input d
    """
    if isinstance(d, str):
        d_str = d
        d = datetime.datetime.strptime(d, DATE_SCHEMA)
    else:
        d_str = d.strftime(DATE_SCHEMA)
    
    # edge case - last or above all
    last_date = dates[-1]
    if not isinstance(last_date, str):
        last_date = dates[-1].strftime(DATE_SCHEMA)
    if d_str >= last_date:
        return len(dates) - 1
    # edge case - first or below all
    first_date = dates[0]
    if not isinstance(first_date, str):
        first_date = first_date.strftime(DATE_SCHEMA)
    if d_str <= first_date:
        return 0
    
    # Perform binary search to find closest entry
    low = 0
    high = len(dates) - 1
    best_diff = np.inf
    best_i = None

    while low < high:
        
        # Check the midpoint
        mid = low + int((high - low)/2)
        d2_str = dates[mid]
        if not isinstance(d2_str, str):
            d2_str = d2_str.strftime(DATE_SCHEMA)

        # We're over if we've found what we're looking for
        if d_str == d2_str:
            return mid
        
        # Continue with the binary search
        elif d_str > d2_str:
            low = mid + 1 
        else:
            high = mid - 1

    # Return the closest if no exact match was found
    d2 = datetime.datetime.strptime(d2_str, DATE_SCHEMA)
    diff2 = (d - d2).days
    if diff2 < 0:
        d3 = dates[mid - 1]
        if isinstance(d3, str):
            d3 = datetime.datetime.strptime(d3, DATE_SCHEMA)
        diff3 = (d - d3).days
        if abs(diff2) < abs(diff3):
            return mid
        else:
            return mid - 1
    else:
        d3 = dates[mid + 1]
        if isinstance(d3, str):
            d3 = datetime.datetime.strptime(d3, DATE_SCHEMA)
        diff3 = (d - d3).days
        if abs(diff2) < abs(diff3):
            return mid
        else:
            return mid + 1
        

def find_closest_date(d: Union[str, datetime.datetime],
                      dates: list) -> int:
    """
    Finds the index of the closest date in
    dates to the date d. If d/dates
    are strings, assumes they are in the default
    date format.

    Tie goes to the earlier index.

    Args:
        d (str or datetime): date to find the closest entry to
        dates (list of str or datetime): list of dates to compare to
    
    Returns:
        int: index of the closest date in dates to the input d
    """

    if isinstance(d, str):
        d = datetime.datetime.strptime(d, DATE_SCHEMA)

    closest_i = 0
    diff = np.inf
    for i in range(len(dates)):
        d2 = dates[i]
        if isinstance(d2, str):
            d2 = datetime.datetime.strptime(d2, DATE_SCHEMA)
        diff2 = abs((d - d2).days)
        if diff2 < diff:
            diff = diff2
            closest_i = i

    return closest_i


def binarySearch(array, x, low, high):

    # Repeat until the pointers low and high meet each other
    while low <= high:

        mid = low + (high - low)//2

        if array[mid] == x:
            return mid

        elif array[mid] < x:
            low = mid + 1

        else:
            high = mid - 1

    return -1



def get_team_ids(sc: OAuth2, league: yfa.league.League) -> pd.DataFrame:
    """
    get the team id, manager, team name, and team object for each team in the league


    Parameters
    ----------
    sc: class
         yahoo oauth object.
    league : class
        yahoo_fantasy_api.league.League.

    Returns
    -------
    teamDF : pandas dataframe
        contains the team id, manager, team name for each team, and team object

    """
    # extract team info from league
    teams = league.teams()
    teamInfo = [[teamID, item['managers'][0]['manager']['nickname'], item['name'], yfa.Team(sc,teamID)]
                for teamID, item in teams.items()]

    # construct dataframe
    teamDF = pd.DataFrame(teamInfo, columns = ['teamID', 'manager','teamName', 'teamObject'])

    return teamDF


def refresh_oauth_file(oauthFile: str = 'yahoo_oauth.json',
                       sport: str = 'nba',
                       year: int = int(DEFAULT_SEASON[:4]),
                       refresh: bool = False) -> (OAuth2,
                                                  yfa.league.League,
                                                  yfa.game.Game):
    """
    refresh the json file with your consumer secret and consumer key 


    Parameters
    ----------
    oauthFile: str, optional
         file path to file with consumer key and consumer secret. The default is 'yahoo_oauth.json'.
    sport : str, optional
        league for the stats you want. The default is 'nba'
    year: int, optional
        year of the league you want. The default is 2022
    refresh: bool, optional
        flag to use if you want to refresh your oauth key. This is done by deleting the other
        variables in the given oauthFile. The default is false.

    Returns
    -------
    sc : class
        yahoo_oauth object.
    gm : class
        nba fantasy group
    currentLeague: class
        league for the given year

    """
    if refresh:
        ext = os.path.splitext(oauthFile)[1]

        # load in the file
        if ext =='.json':
            # read the current json file
            with open(oauthFile, 'r') as f:
                oauthKeys = json.load(f)
        elif ext =='.yaml':
            # read the current json file
            with open(oauthFile, 'r') as f:
                oauthKeys = yaml.safe_load(f)
        else:
            raise ValueError('Wrong file format for yahoo oauth keys. Please use json or yaml')

        # make a new dictionary with the consumer key and consumer secret variables
        newKeys = {}
        newKeys['consumer_key'] = oauthKeys['consumer_key']
        newKeys['consumer_secret'] = oauthKeys['consumer_secret']

        # delete the original json file before writing a new one
        os.remove(oauthFile)

        # save out the new keys to the original file
        with open(oauthFile, 'w') as f:
            if ext =='.json':
                json.dump(newKeys, f)
            elif ext == '.yaml':
                yaml.dump(newKeys, f)

    # set up authenication
    sc = OAuth2(None, None, from_file=oauthFile)

    # get the nba fantasy group
    gm = yfa.Game(sc, sport)

    # get the current nba fantasy league
    league = gm.league_ids(year=year)

    # get the current league stats based on the current year id
    currentLeague = gm.to_league(league[0])

    return sc, gm, currentLeague


def extract_matchup_scores(league: yfa.league.League,
                           week: int,
                           nba_cols: bool = True) -> pd.DataFrame:
    """
    extract the matchup stats for each person for the given week


    Parameters
    ----------
    league : class
         yahoo_fantasy_api.league.League
    week : int
         week to extract matchup data from.
    nba_cols: bool
        add additional columns that have the NBA api stat name

    Returns
    -------
    df : pandas dataframe
        contains matchup stats for each person for a given week.

    """
    # parse the stat categories
    statCats = league.stat_categories()
    statCats = [statNames['display_name'] for statNames in statCats]

    # get the current week
    curWeek = league.matchups(week)['fantasy_content']['league'][1]['scoreboard']['0']['matchups']

    # get each team in the matchup
    matchupStats = []

    # get stats for each matchup
    for matchupNumber in range(curWeek['count']):
        matchupNumber = str(matchupNumber)
        curMatchup = curWeek[matchupNumber]['matchup']['0']['teams']
        for team in range(curMatchup['count']):
            team = str(team)
            teamInfo, teamStats = curMatchup[team]['team']
            teamStats = teamStats['team_stats']['stats']
            # separate the FG/FT count stats
            fg = teamStats[0]['stat']['value'].split('/')
            ft = teamStats[2]['stat']['value'].split('/')
            teamStats = [teamStats[1]] + teamStats[3:]
            labeledStats = {statNames: float(statValues['stat']['value']) if statValues['stat']['value'] else 0
                    for statNames,statValues in zip(statCats,teamStats)}
            if fg[0] == '':
                fg[0] = 0
            if fg[1] == '':
                fg[1] = 0    
            if ft[0] == '':
                ft[0] = 0
            if ft[1] == '':
                ft[1] = 0    
            labeledStats['FGM'] = float(fg[0])
            labeledStats['FGA'] = float(fg[1])
            labeledStats['FTM'] = float(ft[0])
            labeledStats['FTA'] = float(ft[1])
            labeledStats['manager'] = teamInfo[-1]['managers'][0]['manager']['nickname']
            labeledStats['teamName'] = teamInfo[2]['name']
            labeledStats['teamID'] = teamInfo[0]['team_key']
            labeledStats['matchupNumber'] = matchupNumber
            matchupStats.append(labeledStats)

    # once we have all the stats, make a dataframe for the comparison
    df = pd.DataFrame(matchupStats)
    # update the % categories to have more than 3 decimal 
    df['FG%'] = df['FGM']/df['FGA']
    df['FT%'] = df['FTM']/df['FTA']
    df.loc[df['FGA']==0, 'FG%'] = 0
    df.loc[df['FTA']==0, 'FT%'] = 0
    # save the week as the dataframe name
    df.name = week

    # Append NBA API column names
    if nba_cols:
        df['TOV'] = df['TO']
        df['FG3M'] = df['3PTM']
        df['STL'] = df['ST']

    return df

def date_range(date1: str, date2: str):
    dates = []
    start_day = datetime.datetime.strptime(date1, DATE_SCHEMA)
    end_day = datetime.datetime.strptime(date2, DATE_SCHEMA)
    for date in pd.date_range(start_day, end_day, freq='D'):
        dates.append(date.strftime(DATE_SCHEMA))
    return dates


def matchup_winner(stats1, stats2):
    
    counts = [0, 0, 0]
    for stat in ["3PTM", "PTS", "REB", 
                 "AST", "ST", "BLK",
                 "TO", "FG%", "FT%"]:
        if stat != "TO":
            if stats1[stat] > stats2[stat]:
                counts[0] += 1
            elif stats1[stat] < stats2[stat]:
                counts[1] += 1
            else:
                counts[2] += 1
        else:
            if stats1[stat] > stats2[stat]:
                counts[1] += 1
            elif stats1[stat] < stats2[stat]:
                counts[0] += 1
            else:
                counts[2] += 1
    return counts


def calc_fantasy_points(stats):
    return (stats["PTS"] + 1.2*stats["REB"] + 1.5*stats["AST"] + 
            3.0*stats["BLK"] + 3.0*stats["ST"] + -1*stats["TO"])

