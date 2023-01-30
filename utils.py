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
    dates = {}
    for date in pd.date_range(start_date, end_date, freq='D'):
        game_date=date.strftime("%m/%d/%Y")
        try:
            sb = scoreboard.Scoreboard(game_date=game_date)
            teams = sb.get_data_frames()[1]["TEAM_ABBREVIATION"].values
            dates[date] = teams
            for t in teams:
                if t in games_played:
                    games_played[t]+=1
                else:
                    games_played[t] = 1
        except Exception as e:
            print("no games on", game_date)
            raise(e)
    return games_played, dates


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


def get_all_taken_players_extra(sc, league, week, include_today=False, actual_played=False):
    """
    Returns a dictionary of all taken players, with entries appended
    for:
        1) which NBA team they're on
        2) which fantasy team they're on 
        3) what position they are currently placed on for fantasy (to check if they're on IL)
        4) how many games they have played already this week (does not count games today by default)
        5) how many games total they have on the calendar for this week
        6) nba api name
    
    Args:
        sc: yahoo oauth object
        league: yahoo api league object
        week: int yahoo week to return number of games for
        include_today: include today as games played

    """
    cur_wk = league.current_week()


    # Get games played information for nba teams
    this_week, teams_playing = num_games_played_per_week(league, week)
    d0, df = league.week_date_range(week)
    if d0 <= TODAY <= df:
        if include_today:
            up_today = num_games_played(d0, TODAY)[0]
        else:
            up_today = num_games_played(d0, TODAY - datetime.timedelta(days=1))[0]
    else:
        up_today = {}
    for t in this_week:
        if t not in up_today:
            up_today[t] = 0

    # Get player logs to determine nba team
    logs = get_all_player_logs()

    # get the roster for all the teams in the fantasy league for today
    # key is player name goes to teamID, manager, teamName, status, position_type, eligible_positions, and selected_position
    tp = [] # list of all taken players
    teamDF = get_team_ids(sc, league)
    actual_num_played = {}

    for i,row in teamDF.iterrows():
        manager = row['manager']
        # If we're looking at the past
        if week != cur_wk or (TODAY==df and include_today):
            if actual_played:
                for date in pd.date_range(d0, df, freq='D'):
                    team_roster = row['teamObject'].roster(day=date)
                    for p in team_roster:
                        pos = p['selected_position']
                        status = p['status']
                        name = yahoo_to_nba_name(p['name'])
                        nba_team = get_nba_team(name, logs)

                        # Catches players that have never played this year
                        # Looking at you Chi Yen/Lonzo Ball
                        try:
                            player_stats = logs.get_group(name)
                        except:
                            continue

                        played = ("IL" not in pos) and ("BN" not in pos) and (nba_team in teams_playing[date])
                        if played:
                            # Now check to see if they had any stats -- to show if they missed
                            # due to injury or gtd
                            game_stats = player_stats[date.strftime("%Y-%m-%dT00:00:00") == player_stats.GAME_DATE]
                            if game_stats.shape[0] < 1:
                                # print(p, "missed on", date)
                                continue
                            elif game_stats.iloc[0]['MIN'] == 0:
                                # print(p, "missed on", date)
                                continue
                            if p['name'] not in actual_num_played:
                                actual_num_played[p['name']] = {"date":[date.date()], 
                                                        "manager":[manager],
                                                        "teamName":[row['teamName']],
                                                        "name":p['name'],
                                                        "selected_position":pos,
                                                        "eligible_positions": p['eligible_positions'],
                                                        "status": status}
                            else:
                                actual_num_played[p['name']]['date'] += [date.date()]
                                actual_num_played[p['name']]['manager'] += [manager]
                                actual_num_played[p['name']]['teamName'] += [row['teamName']]

            else:
                team_roster = row['teamObject'].roster(day=d0)
        else:
            team_roster = row['teamObject'].roster()

        if not actual_played:
            for p in team_roster:
                name = p['name']
                info = {}
                for k in p:
                    info[k] = p[k]
                p['teamID'] = row['teamID']
                p['manager'] = manager
                p['teamName'] = row['teamName']
                tp.append(p)

    if actual_played:
        for p in actual_num_played:
            tp.append(actual_num_played[p])
    
    players = {}
    for p in tp:

        name = yahoo_to_nba_name(p['name'])
        entry = {}
        entry['name'] = p['name']
        entry['nba_name'] = name
        entry['nba_team'] = get_nba_team(name, logs)
       

        entry['games_total'] = this_week[entry['nba_team']]
        entry['games_played'] = up_today[entry['nba_team']]
        entry['manager'] = p['manager']
        entry['fantasy_team'] = p['teamName']
        entry['selected_position'] = p['selected_position']
        entry['eligible_positions'] = p['eligible_positions']
        entry['status'] = p['status']
        if actual_played:
            entry['actual_played'] = actual_num_played[entry['name']]
        players[name] = entry

    return players

def get_nba_team(nba_name, logs):

    # Catch Inactive all year players
    try:
        team_name = logs.get_group(nba_name)["TEAM_ABBREVIATION"].iloc[0]
    except:
        # Hard code never played CHET
        if nba_name == 'Chet Holmgren':
            team_name = 'OKC'
        else:
            player_id = find_players_by_full_name(nba_name)[0]['id']
            career = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0].sort_values("PLAYER_AGE")
            team_name = career.iloc[-1]["TEAM_ABBREVIATION"][-3:]
    
    return team_name

    
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


def extract_matchup_scores(league, week, nba_cols = True):
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

if __name__ == "__main__":
    sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')
    tp = get_all_taken_players_extra(sc, curLg, curLg.current_week()-1, actual_played=True)