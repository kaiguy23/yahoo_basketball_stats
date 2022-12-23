import os
import json
import yaml 
import math
import numpy as np
import pandas as pd
import datetime

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa


from nba_api.stats.endpoints import playergamelogs, playercareerstats
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
    games_played = {}
    for date in pd.date_range(start_date, end_date, freq='D'):
        game_date=date.strftime("%m/%d/%Y")
        try:
            sb = scoreboard.Scoreboard(game_date=game_date)
            teams = sb.get_data_frames()[1]["TEAM_ABBREVIATION"].values
            for t in teams:
                if t in games_played:
                    games_played[t]+=1
                else:
                    games_played[t] = 1
        except:
            print("no games on", game_date)
    return games_played


def num_games_played_per_week(league, week):
    """
    Returns a dictionary that says how many games each nba 
    team plays in each given week of the league

    Args:
        league: yahoo api league object
    """
    # Build dictionary of teams and how many games they've played
    start_date, end_date = league.week_date_range(week)
    return num_games_played(start_date, end_date)


def get_all_taken_players_extra(sc, league, week):
    """
    Returns a dictionary of all taken players, with entries appended
    for:
        1) which NBA team they're on
        2) which fantasy team they're on 
        3) what position they are currently placed on for fantasy (to check if they're on IL)
        4) how many games they have played already this week
        5) how many games total they have on the calendar for this week
        6) nba api name
    
    Args:
        sc: yahoo oauth object
        league: yahoo api league object
        week: int yahoo week to return number of games for

    """

    # Get all the players currently on teams
    tp = league.taken_players()

    # Get games played information for nba teams
    this_week = num_games_played_per_week(league, week)
    if TODAY > league.week_date_range(week)[0]:
        up_today = num_games_played(league.week_date_range(week)[0], TODAY)
    else:
        up_today = {}
    for t in this_week:
        if t not in up_today:
            up_today[t] = 0

    # get the roster for all the teams in the fantasy league for today
    # key is player name goes to teamID, manager, teamName, status, position_type, eligible_positions, and selected_position
    rosters = {}
    teamDF = get_team_ids(sc, league)
    for i,row in teamDF.iterrows():
        team_roster = row['teamObject'].roster()
        for p in team_roster:
            name = p['name']
            info = {}
            for k in p:
                info[k] = p[k]
            info['teamID'] = row['teamID']
            info['manager'] = row['manager']
            info['teamName'] = row['teamName']
            rosters[name] = info

    # Get player logs to determine team
    logs = get_all_player_logs()
    players = {}
    for p in tp:
        name = yahoo_to_nba_name(p['name'])
        entry = {}
        entry['yahoo_name'] = p['name']
        entry['nba_name'] = name

        # Catch Inactive all year players
        try:
            entry['nba_team'] = logs.get_group(name)["TEAM_ABBREVIATION"].iloc[0]
        except:
            # Hard code never played CHET
            if name == 'Chet Holmgren':
                entry['nba_team'] = 'OKC'
            else:
                player_id = find_players_by_full_name(name)[0]['id']
                career = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0].sort_values("PLAYER_AGE")
                entry['nba_team'] = career.iloc[-1]["TEAM_ABBREVIATION"][-3:]

        entry['games_total'] = this_week[entry['nba_team']]
        entry['games_played'] = up_today[entry['nba_team']]
        entry['manager'] = rosters[p['name']]['manager']
        entry['fantasy_team'] = rosters[p['name']]['teamName']
        entry['selected_position'] = rosters[p['name']]['selected_position']
        entry['eligible_positions'] = rosters[p['name']]['eligible_positions']
        entry['status'] = rosters[p['name']]['status']
        players[name] = entry

    return players


    
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


if __name__ == "__main__":
    sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')
    tp = get_all_taken_players_extra(sc, curLg, curLg.current_week())