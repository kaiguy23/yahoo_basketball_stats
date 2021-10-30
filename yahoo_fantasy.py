import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sn
import pprint

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

    if visualize:
        # make save directory
        weekSaveDir = os.path.join(saveDir, f'week{df.name}')
        os.makedirs(weekSaveDir,exist_ok=True)

        # save out the dataframe
        df.to_csv(os.path.join(weekSaveDir,'matchupTotals.csv'),index=False)

        print(df[['manager', 'teamName','totalWins','totalLosses']])

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

def max_min_stats(league, df, visualize=True):
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
    statCats = league.stat_categories()
    statCats = [statNames['display_name'] for statNames in statCats]
    statCats += ['FGM', 'FGA', 'FTM', 'FTA']

    # also compute the highest/lowest per category
    maxStatDict = {s:{} for s in statCats}
    maxCats = df.loc[df[statCats].idxmax()]
    maxCats.reset_index(drop=True,inplace=True)
    for idx, row in maxCats.iterrows():
        curCat = statCats[idx]
        maxStatDict[curCat] = {row['manager']: row[curCat]}
    
    minStatDict = {s:{} for s in statCats}
    minCats = df.loc[df[statCats].idxmin()]
    minCats.reset_index(drop=True,inplace=True)
    for idx, row in minCats.iterrows():
        curCat = statCats[idx]
        minStatDict[curCat] = {row['manager']: row[curCat]}
    
    if visualize:
        pprint.pprint('Lowest Stat Totals')
        pprint.pprint(minStatDict)
        
        pprint.pprint('Highest Stat Totals')
        pprint.pprint(maxStatDict)

    return maxStatDict, minStatDict

if __name__ == '__main__':

    # set up authentication
    sc = OAuth2(None, None, from_file='yahoo_oauth.json')

    # get the nba fantasy group
    gm = yfa.Game(sc, 'nba')

    # get the current nba fantasy league
    lg2021 = gm.league_ids(year=2021)

    # get the current league stats based on the current year id
    curLg2021 = gm.to_league(lg2021[0])

    # for each previous week (don't include the current one)
    # yahoo week index starts at 1 so make sure to start looping at 1
    for week in range(1,curLg2021.current_week()):
        # get the current week matchup stats
        df = extract_matchup_scores(curLg2021, week)
        # calculate matchups
        df, matchupScore, matchupWinner = create_matchup_comparison(curLg2021, df)
        # also compute the highest/lowest per category
        maxStatDict, minStatDict = max_min_stats(curLg2021, df)