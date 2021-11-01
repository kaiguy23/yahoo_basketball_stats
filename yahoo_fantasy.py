import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import datetime
import seaborn as sn
import pprint
import json
import yaml
import dataframe_image as dfi

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

def extract_matchup_scores(league, week):
    """
    extract the matchup stats for each person for the given week
    inputs:
        league: class, yahoo_fantasy_api.league.League
        week: int, week to extract matchup data from
    returns:
        df: pandas dataframe, contains matchup stats for each person for a given week
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
            labeledStats = {statNames:float(statValues['stat']['value'])
                        for statNames,statValues in zip(statCats,teamStats)}
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

    # save the week as the dataframe name
    df.name = week
    return df

def create_matchup_comparison(league, df, visualize=True, saveDir='matchup results'):
    """
    create the matchup matrix to compare each person to each other
    inputs:
        league: class, yahoo_fantasy_api.league.League
        df: pandas dataframe, contains matchup stats for each person for a given week
    returns:
        df: pandas dataframe, contains matchup stats for each person for a given week
        matchupScore: numpy array, contains the matchup score for each head to head comparison
        matchupWinner: numpy array, winner of the head to head matchups
    """
    # parse the stat categories
    statCats = league.stat_categories()
    statCats = [statNames['display_name'] for statNames in statCats]

    # split up the categories into TOs and non TOs
    nonTOs = statCats[:-1]
    TOs = statCats[-1]

    # create arrays for the matchup score and the matchup winner
    matchupScore = np.empty((df.shape[0],df.shape[0]),dtype=object)
    matchupWinner = np.zeros((df.shape[0],df.shape[0]))
    for i in range(df.shape[0]):
        curTeam = df.loc[i]
        for j in range(df.shape[0]):
            # compare the rows aka matchups
            vsTeam = df.loc[j]

            # compute score of the curTeam against the vs Team
            curScore = sum(curTeam[nonTOs]>vsTeam[nonTOs]) + int(curTeam[TOs]<vsTeam[TOs])
            vsScore = len(statCats) - curScore
            matchupScore[i,j] = f'{curScore}-{vsScore}'
            matchupScore[j,i] = f'{vsScore}-{curScore}'
            matchupWinner[i,j] = curScore > vsScore
            matchupWinner[j,i] = vsScore > curScore

    # get diagonal indices and turn them off
    idxs = np.diag_indices(df.shape[0])

    # 0 them out before doing the biggest winners / losers
    matchupWinner[idxs] = 0
    df['totalWins'] = matchupWinner.sum(axis=1)

    # subtract 1 since you can't play yourself
    df['totalLosses'] = df.shape[0] - 1 - df['totalWins']

    # set the diagonals idxs
    matchupWinner[idxs] = -1
    matchupScore[idxs] = 'N/A'

    # fix manager names and team names for saving out data
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

    if visualize:
        # make save directory
        weekSaveDir = os.path.join(saveDir, f'week{df.name}')
        os.makedirs(weekSaveDir,exist_ok=True)

        # save out the dataframe
        df.to_csv(os.path.join(weekSaveDir,'matchupTotals.csv'),index=False)
        dfi.export(df[['manager', 'teamName','totalWins','totalLosses']], os.path.join(weekSaveDir,'matchupTotalsTable.png'))

        # set colors so that black = -1 / N/A, red = 0 / Loss, green = 1 / Win
        colors = ['black','red','green']
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        # create labels for the axes
        yAxisLabels = df[['manager', 'teamName']].apply(lambda x: x[0] + '\n' + x[1],axis=1)
        xAxisLabels = df['manager']

        # do plotting
        sn.set(font_scale=1.2)
        f, ax = plt.subplots(figsize=(20,10))
        ax = sn.heatmap(matchupWinner, annot=matchupScore, fmt='', xticklabels = xAxisLabels,
                yticklabels = yAxisLabels, cmap=cmap,cbar=False)

        # add in patches to mark who actually played who in that week
        # get number of unique matchups:
        for m in df['matchupNumber'].unique():
            i,j = df[df['matchupNumber']==m].index
            ax.add_patch(Rectangle((i,j), 1, 1, fill=False, edgecolor='blue', lw=3))
            ax.add_patch(Rectangle((j,i), 1, 1, fill=False, edgecolor='blue', lw=3))
        f.suptitle(f'NBA Fantasy Week {week} Matchups', fontsize = 30)
        ax.set_title('Blue boxes indicate the actual matchups', fontsize = 15)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(weekSaveDir,'matchups.png'))

    return df, matchupScore, matchupWinner

def max_min_stats(league, df, visualize=True, saveDir='matchup results'):
    """
    calculate the min and max stats for each category for the week
    inputs:
        league: class, yahoo_fantasy_api.league.League
        df: pandas dataframe, contains matchup stats for each person for a given week

    returns:
        maxStatDict: dictionary with highest stat totals for the week
        minStatDict: dictionary with lowest stat totals for the week
    """
    # parse the stat categories
    origStatCats = league.stat_categories()
    origStatCats = [statNames['display_name'] for statNames in origStatCats]
    statCats = origStatCats + ['FGM', 'FGA', 'FTM', 'FTA']

    # make save directory
    weekSaveDir = os.path.join(saveDir, f'week{df.name}')
    os.makedirs(weekSaveDir,exist_ok=True)

    # compute the highest per category
    maxStatList = []
    maxCats = df.loc[df[statCats].idxmax()]
    maxCats.reset_index(drop=True,inplace=True)
    for idx, row in maxCats.iterrows():
        curCat = statCats[idx]
        maxStatList.append([curCat, row[curCat], row['manager']])

    # save out the data
    maxStatDF = pd.DataFrame(maxStatList, columns = ['Stat', 'Value', 'Manager'])
    maxStatDF.to_csv(os.path.join(weekSaveDir,'maxStats.csv'),index=False)

    # compute lowest per category
    minStatList = []
    minCats = df.loc[df[statCats].idxmin()]
    minCats.reset_index(drop=True,inplace=True)
    for idx, row in minCats.iterrows():
        curCat = statCats[idx]
        minStatList.append([curCat, row[curCat], row['manager']])

    # save out the data
    minStatDF = pd.DataFrame(minStatList, columns = ['Stat', 'Value', 'Manager'])
    minStatDF.to_csv(os.path.join(weekSaveDir,'minStats.csv'),index=False)

    if visualize:
        dfi.export(minStatDF, os.path.join(weekSaveDir,'minStats.png'))
        dfi.export(maxStatDF, os.path.join(weekSaveDir,'maxStats.png'))

        # make bar charts as well
        # sorted stats
        fig, ax = plt.subplots(3,3, figsize = (20,20))
        for idx, s in enumerate(origStatCats):
            row = idx // 3
            col = idx % 3
            df.sort_values(by = s, inplace = True)
            ax[row,col].bar(df['manager'], df[s])
            ax[row,col].set_title(s)
        plt.suptitle('Sorted Stats')
        plt.savefig(os.path.join(weekSaveDir,'sortedStats.png'))
        plt.close()

        # sorted by manager name
        df.sort_values(by = 'manager', inplace = True)
        fig, ax = plt.subplots(3,3, figsize = (20,20))
        for idx, s in enumerate(origStatCats):
            row = idx // 3
            col = idx % 3
            ax[row,col].bar(df['manager'], df[s])
            ax[row,col].set_title(s)
        plt.suptitle('Stats')
        plt.savefig(os.path.join(weekSaveDir,'stats.png'))
        plt.close()

    return maxStatDF, minStatDF

def get_team_ids(sc, league):
    """
    get the team id, manager, team name, and team object for each team in the league
    inputs:
        sc: class, yahoo oauth object
        league: class, yahoo_fantasy_api.league.League
    returns:
        teamDF: pandas dataframe, contains the team id, manager, team name for each team, and team object
    """
    # extract team info from league
    teams = league.teams()
    teamInfo = [[teamID, item['managers'][0]['manager']['nickname'], item['name'], yfa.Team(sc,teamID)]
                for teamID, item in teams.items()]

    # construct dataframe
    teamDF = pd.DataFrame(teamInfo, columns = ['teamID', 'manager','teamName', 'teamObject'])

    return teamDF

def refresh_oauth_file(oauthFile = 'yahoo_oauth.json', sport = 'nba', year = 2021, refresh = False):
    """
    refresh the json file with your consumer secret and consumer key by deleting the other variables.
    this is done to avoid the yahoo api max call limit, which will just give you a request denied.
    you will have to re-enter the yahoo key that will opened in an internet browser

    inputs:
        jsonFile: json, file path to file with consumer key and consumer secret
        sport: str, league for the stats you want
        year: int, year of the league you want
    returns:
        sc: yahoo_oauth, key for yahoo api
        gm: class, nba fantasy group
        currentLeague: class, league for the given year
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

if __name__ == '__main__':

    sc, gm, curLg2021 = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')

    # for each previous week (don't include the current one)
    # yahoo week index starts at 1 so make sure to start looping at 1
    for week in range(1,curLg2021.current_week()):
        # get the current week matchup stats
        df = extract_matchup_scores(curLg2021, week)
        # calculate matchups
        df, matchupScore, matchupWinner = create_matchup_comparison(curLg2021, df)
        # also compute the highest/lowest per category
        maxStatDF, minStatDF = max_min_stats(curLg2021, df)

        # add in continue here so the code doesn't break from the new code added below.
        # currently getting hit with a "RuntimeError: b'Request denied\r\n'"
        # I think I'm requesting too much data from their servers... RIP
        continue

        teamDF = get_team_ids(sc, curLg2021)
        startDate, endDate = curLg2021.week_date_range(week)
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
        # get stat categories we care about
        statCats = curLg2021.stat_categories()
        statCats = [statNames['display_name'] for statNames in statCats]
        for currentDate in dateRanges:
            # get the roster for the current date
            teamDF[currentDate] = teamDF['teamObject'].apply(lambda teamObject: pd.DataFrame(teamObject.roster(day = currentDate)))
            # loop through the roster for each team. roster is the same thing as teamDF.loc[idx,currentDate]
            for idx, roster in enumerate(teamDF[currentDate]):
                # set if the player was started
                teamDF.loc[idx,currentDate]['started'] = roster['selected_position'].apply(lambda pos: True if pos != 'IL' and pos != 'IL+' and pos != 'BN' else False)
                # grab the player stats for the whole roster
                playersStats = roster['player_id'].apply(lambda player_id: curLg2021.player_stats(player_id, 'date', date = currentDate)).to_list()
                # loop through each player in the roster and assign their stats to the roster dataframe
                for player in playersStats:
                    if len(player) != 1:
                        print('Only one list of stats should be returned for each player.')
                        print('Using the first item in the list')
                    # the league.player_stats function returns a list of dictionaries. There should only be one for each player
                    player = player[0]
                    # for each stat in the league stat categories, save out the stats
                    for currentStat in statCats:
                        teamDF.loc[idx,currentDate].loc[roster['player_id']==player['player_id'],currentStat] = 0 if player[currentStat] == '-' else player[currentStat]
        # also calculate the stats for each team if they kept their team from last week
        for idx, roster in enumerate(teamDF[previousSunday]):
            # initiate each stat at 0 for each player in the sunday roster for the current team
            for currentStat in statCats:
                teamDF.loc[idx,previousSunday][currentStat] = 0
            # for each day in the week get the players stats
            for currentDate in dateRanges:
                # grab the player stats for the whole roster
                playersStats = roster['player_id'].apply(lambda player_id: curLg2021.player_stats(player_id, 'date', date = currentDate)).to_list()
                # loop through each player in the roster and assign their stats to the roster dataframe
                for player in playersStats:
                    if len(player) != 1:
                        print('Only one list of stats should be returned for each player.')
                        print('Using the first item in the list')
                    # the league.player_stats function returns a list of dictionaries. There should only be one for each player
                    player = player[0]
                    # for each stat in the league stat categories, save out the stats
                    for currentStat in statCats:
                        teamDF.loc[idx,previousSunday].loc[roster['player_id']==player['player_id'],currentStat] += 0 if player[currentStat] == '-' else player[currentStat]
        # TODO: Figure out which team had the best GM
        # Figure out which players on each roster was added and dropped over the course of the week for each team
        # Just compare back to the sunday roster.