import os
import json
import yaml 
import math
import numpy as np
import pandas as pd
import datetime

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa


from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static.players import find_players_by_first_name, find_players_by_full_name, find_players_by_last_name
from nba_api.stats.endpoints import scoreboard

# Get the default season
TODAY = datetime.date.today()
if TODAY.month > 7:
    DEFAULT_SEASON = f"{TODAY.year}-{str(TODAY.year+1)[-2:]}"
else:
    DEFAULT_SEASON = f"{TODAY.year-1}-{str(TODAY.year)[-2:]}"


SPECIAL_NAMES = {}
def yahoo_to_nba_name(name, hardcoded = SPECIAL_NAMES):
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


def num_games_played(start_date, end_date):
    """
    Returns a dictionary that says how many games each 
    team plays between a start and end date (both sides inclusive)

    Args:
        start_date (datetime obj): start date
        end_date (datetime obj): end date
    """
    # Build dictionary of teams and how many games they've played
    for date in pd.date_range(start_date, end_date, freq='D'):

        d = scoreboard.Scoreboard(game_date=date.strftime("DD/MM/YYYY"))
        d.get_data_frames()[1]["TEAM"]


def get_all_taken_players_extra():
    """
    Returns a list of all taken players, with entries appended
    for:
        1) which NBA team they're on
        2) which fantasy team they're on 
        3) what position they are currently placed on for fantasy (to check if they're on IL)
        4) how many games they have remaining in each of the remaining weeks
    """

    # Get all the players currently on teams
    tp = league.taken_players()


def get_all_player_logs(season=DEFAULT_SEASON):
    """
    Returns a pandas groupby object that maps player name to
    a log of all their game stats, sorted by game date with most recent first

    Args:
        season (str, optional): season in the format like 2022-23
    """
    stats = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    return stats.sort_values(by="GAME_DATE",ascending=False).groupby("PLAYER_NAME")

def fix_names_teams(df):
    """
    edits the dataframe's teamName and manager columns so we can save them our correctly
    

    Parameters
    ----------
    df : pandas dataframe
        contains teamName and manager columns.

    Returns
    -------
    df : pandas dataframe
        modified dataframe.

    """
    for idx, row in df.iterrows():
        manager = row['manager']
        team = row['teamName']
        newManager = ''
        newTeam = ''
        for char in manager:
            if char.isalnum() or char == ' ' or char == '-':
                newManager += char
        for char in team:
            if char.isalnum() or char == ' ' or char == '-':
                newTeam += char
        df.loc[idx, 'manager'] = newManager
        df.loc[idx, 'teamName'] = newTeam
    return df

def get_team_ids(sc, league):
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


def refresh_oauth_file(oauthFile = 'yahoo_oauth.json', sport = 'nba', year = 2022, refresh = False):
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

