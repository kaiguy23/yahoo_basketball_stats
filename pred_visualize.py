from pathlib import Path
import os
from tqdm import tqdm

from scipy.stats import skellam, poisson, norm

import numpy as np
import pandas as pd
import seaborn as sn

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pred, utils
from db_interface import dbInterface
from db_builder import dbBuilder

def plot_matchup_history(db, week, p1, p2, savename = None):

    team1 = db.manager_to_teamID(p1)
    team2 = db.manager_to_teamID(p2)
    if savename is None:
        savename = f"{p1}_vs_{p2}_week_{week}_history.png"

    res = {"Overall":[]}
    proj_dict = {}
    preds_dict = {}
    date_range = utils.date_range(db.week_date_range(week)[0], db.week_date_range(week)[1])
    for date in date_range:
        proj = pred.proj_all_players(db, date)
        preds = pred.predict_matchup(db, date, team1, team2, proj=proj)
        res["Overall"].append(preds[0])
        for stat in preds[1]:
            if stat not in res:
                res[stat] = []
            res[stat].append(np.array(preds[1][stat][0]))
        proj_dict[date] = proj
        preds_dict[date] = preds
    
    final_scoreboard = db.matchup_score(week)
    for stat in res:
        res[stat] = np.vstack(res[stat])
        if stat == "Overall":
            winners = utils.matchup_winner(final_scoreboard.loc[team1], final_scoreboard.loc[team2])[0]
        else:
            winners = [final_scoreboard.loc[team1][stat], final_scoreboard.loc[team2][stat]]
        if stat != "TO":
            if winners[0] > winners[1]:
                res[stat] = np.vstack((res[stat], np.array([1,0,0])))
            elif winners[0] < winners[1]:
                res[stat] = np.vstack((res[stat], np.array([0,1,0])))
            else:
                res[stat] = np.vstack((res[stat], np.array([0,0,1])))
        else:
            if winners[0] > winners[1]:
                res[stat] = np.vstack((res[stat], np.array([0,1,0])))
            elif winners[0] < winners[1]:
                res[stat] = np.vstack((res[stat], np.array([1,0,0])))
            else:
                res[stat] = np.vstack((res[stat], np.array([0,0,1])))
    
    
    xlabels = [d[6:] for d in date_range] + ["Final"]
    # fig, axd = plt.subplot_mosaic([["Overall", "Overall", "Overall", "Perf", "Perf"],
    #                                ["FG%", "FT%", "3PTM", "Perf", "Perf"],
    #                                ["PTS", "REB", "AST", "Perf", "Perf"],
    #                                ["ST", "BLK", "TO", "Perf", "Perf"]],
    #                           figsize=(12, 12), layout="constrained")
    fig, axd = plt.subplot_mosaic([["Overall", "Overall", "Overall"],
                                   ["FG%", "FT%", "3PTM"],
                                   ["PTS", "REB", "AST"],
                                   ["ST", "BLK", "TO"],
                                   ["Perf", "Perf", "Perf"],
                                   ["Perf", "Perf", "Perf"]],
                              figsize=(12, 18), layout="constrained")
    
    fig.suptitle(f"{p1} vs. {p2}\nPredicted Results from Morning of Specified Day", fontsize=20)
    for stat in res:
        # for k in axd:
        #     annotate_axes(axd[k], f'axd["{k}"]', fontsize=14)
        ax = axd[stat]
        ax.plot(res[stat][:,0], label=db.teamID_lookup(team1)[0], marker=".", lw=1)
        ax.plot(res[stat][:,1], label=db.teamID_lookup(team2)[0], marker=".", lw=1)
        ax.plot(res[stat][:,2], label="Tie", marker=".", lw=1)
        ax.set_title(stat, fontsize=14)
        if stat == "Overall":
            ax.set_title(stat, fontsize=16)
            ax.set_ylabel("Probability", fontsize=16)
            ax.legend(fontsize=16)
            ax.set_xticks(ticks=np.arange(len(xlabels)), labels=xlabels, fontsize=16)
        else:
            ax.set_xticks(ticks=np.arange(len(xlabels)), labels=[""]*len(xlabels))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
    # plt.tight_layout()

    # Plot the impactful performances
    # ef = player_effects(db, week, p1, p2)[["name", "manager", "date"] + list(res.keys())]
    ef = player_effects(db, week, p1, p2)
    num_cols = np.array([x for x in res.keys() if x != "Overall"])
    thresh = 0.05
    biggest_cat = []
    for i, row in ef.iterrows():
        good = row["Overall"] >= 0
        to_add = []
        for c in num_cols:
            if good:
                if row[c] > thresh:
                    to_add.append(c)
            else:
                if -row[c] > thresh:
                    to_add.append(c)
        biggest_cat.append(", ".join(to_add))
    ef["big stats"] = biggest_cat
    order = np.argsort(np.abs(ef.Overall.values))[::-1]
    ef[r"$\Delta$ $P_{win/tie}$"] = [str(int(np.round(x,2)*100)) + "%" for x in ef["Overall"].values]
    ef = ef[["manager", "name", "date", r"$\Delta$ $P_{win/tie}$", "big stats"]]
    ef = ef.iloc[order].iloc[:15]
    
    # for col in ef.columns:
    #     if col not in ["name", "manager", "date", "Big Stats"]:
    #         ef[col] = np.round(ef[col], 2)
    ax = axd["Perf"]
    bbox=[0, 0, 1, 1]
    mpl_table = ax.table(cellText=ef.values, bbox = bbox, colLabels=ef.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(14)
    header_color='#40466e'
    row_colors=['#f1f1f2', 'w']
    edge_color='w'
    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    ax.set_yticks([])
    ax.set_xticks([])
    


    plt.savefig(savename)


def plot_matchup_matrix(pred_mat: np.array, order: list[str],
                        matchup_df: pd.DataFrame = None,
                        savename: str = "pred_mat.png"):
    """
    Generates a matrix of predicted outcomes if every player played every other player

    Args:
        pred_mat (np.array): array of probabilities of victory
        order (str): list of manager names
        matchup_df (pd.Dataframe): matchups for the week from db.matchup
        savename (str): file to save the figure to


    Returns:
        None
    """

    yAxisLabels = order
    xAxisLabels = order

    # do plotting
    sn.set(font_scale=1.2)
    f, ax = plt.subplots(figsize=(20,10))
    ax = sn.heatmap(pred_mat, annot=np.round(pred_mat*100)/100, fmt='', xticklabels = xAxisLabels,
            yticklabels = yAxisLabels, cmap='RdYlGn',cbar=False)

    # highlight actual matchup
    if not matchup_df is None:
        # add in patches to mark who actually played who in that week
        # get number of unique matchups:
        for m in matchup_df['matchupNumber'].unique():
            teams = matchup_df[matchup_df['matchupNumber']==m].manager.values
            i, j = order.index(teams[0]), order.index(teams[1])
            print("matchup:", m, "index:", i, j)
            ax.add_patch(Rectangle((i,j), 1, 1, fill=False, edgecolor='blue', lw=3))
            ax.add_patch(Rectangle((j,i), 1, 1, fill=False, edgecolor='blue', lw=3))

    if matchup_df is None:
            f.suptitle(f"NBA Fantasy Predicted Results", fontsize = 30)
    else:
        f.suptitle(f"NBA Fantasy Predicted Results (Week {matchup_df.iloc[0].week})", fontsize = 30)
   
    if savename != "":
        plt.savefig(savename)
        plt.close(f)


# def predict_matchup(db: dbInterface, date: str, team1: str, team2: str, 
#                     proj: pd.DataFrame = None, scores: pd.DataFrame = None,
#                     kern_sig: float = GKERN_SIG, actual_played: bool = False):
    
def plot_matchup_summary(db: dbInterface, date: str, proj: pd.DataFrame, p1: str, p2: str,
                         matchup_df: pd.DataFrame = None, savename: str = None):
    """
    Plots a summary of the specified matchup, showing the 90% confidence interval
    for each stat, showing the probability that either player will 
    win the category.


    Args:
        db (dbInterface): dbInterface object
        date (str): Date in YYYY-MM-DD format
        proj (dict): dictionary that maps player to (projected stats, current stats)
        p1 (str): name of the first player
        p2 (str): name of the second player
        matchup_df (pandas df, optional): matchup df with the current score for midweek analysis. Defaults to None.
        savename (str): file to save the matchup summary to
     """

    probs, stat_victory, outcomes, proj_stats = pred.predict_matchup(db, date, db.manager_to_teamID(p1),
                                                                     db.manager_to_teamID(p2), proj,
                                                                     scores = matchup_df)
    
    f, ax = plt.subplots(nrows=3,ncols=3, figsize=(20,14))

    vic = [np.round(x*100,2) for x in probs]
    outcomes_df = []
    for score in outcomes:
        entry = {}
        entry["score"] = score
        entry["prob"] = outcomes[score]
        outcomes_df.append(entry)
    outcomes_df = pd.DataFrame(outcomes_df).sort_values(by="prob", ascending=False)
    title_str =  f"Probability of Victory - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}% \n"
    title_str += f"Most Probable Outcomes - {outcomes_df.iloc[0]['score']}: {int(np.round(outcomes_df.iloc[0]['prob']*100))}%, "
    title_str += f"{outcomes_df.iloc[1]['score']}: {int(np.round(outcomes_df.iloc[1]['prob']*100))}%, "
    title_str += f"{outcomes_df.iloc[2]['score']}: {int(np.round(outcomes_df.iloc[2]['prob']*100))}%"
    f.suptitle(title_str, fontsize=26)

    # epsilon = 0.0001
    epsilon = 0.05

    if not matchup_df is None:
        grouped = matchup_df.groupby("manager")

    
   
    for ip, p in enumerate((p1,p2)):
        teamID = db.manager_to_teamID(p)
        for i, stat in enumerate(stat_victory):
            if "%" in stat:
                mu_total = stat_victory[stat][ip+1][0]
                sigma = stat_victory[stat][ip+1][1]
                x_err = norm.ppf(1-epsilon,loc=0, scale=sigma)
            else:
                mu_total = stat_victory[stat][ip+1][0]
                sigma = stat_victory[stat][ip+1][1]
                x_err = poisson.ppf(1-epsilon, sigma)
                if x_err == 0:
                    x_err = 1
            row = int(i/3)
            col = i-int(i/3)*3
            a = ax[row,col]

            if ip == 0:
                vic = np.array([np.round(x*100,2) for x in stat_victory[stat][0]])
                a.set_title(f"{stat} - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%")

            a.errorbar(mu_total, (1-ip)*0.01, xerr=np.array((x_err, x_err)).reshape(2,1),
                       label=p, capsize = 4, fmt = 'o', alpha=0.75)
            a.set_ylim([-0.005,0.015])
            a.set_yticks([])
            if ip == 1:
                a.legend()
    
    if savename is None:
        savename = f"{p1}_vs_{p2}.png"
    plt.savefig(savename)

    return


def run_predictions(db: dbInterface, week: int, folder: str, date: str = ""):
    """
    Does the weekly prediction/figure generation
    for the next week.

    Args:
        db (dbInterface): dbInterface object
        week (int): fantasy week to do predictions for
        folder (str): folder to save figures to
        date (str): YYYY-DD-MM to do predictions from the morning
                    of. If none is given, then assumes it's the
                    beginning of the specified week.
    """

    d0, df = db.week_date_range(week)
    if date == "":
        date = d0
    elif d0 <= date <= df:
        d0 = date
    elif date > utils.TODAY_STR:
        date = utils.TODAY_STR
    print("Doing Predictions for Rosters Morning of", date, "with matchups from week", week)
    try:
        matchup_df = db.matchup_score(week, d0)
    except:
        print("Matchups can't be found for week", week)

    proj = pred.proj_all_players(db, date)
    pred_mat, order, stats = pred.matchup_matrix(db, date, actual_played=False)
    
    # Make a matrix plot of the whole league
    plot_matchup_matrix(pred_mat, order,
                        savename=Path(folder,"pred_mat.png"),
                        matchup_df=matchup_df)
    
    # Make a detailed figure for each matchup    
    grouped = matchup_df.groupby("matchupNumber")
    for i in grouped.indices:
        matchup = grouped.get_group(i)
        p1 = matchup['manager'].iloc[0]
        p2 = matchup['manager'].iloc[1]
        savename = str(Path(folder,f"{p1}_vs_{p2}.png"))
        plot_matchup_summary(db, date, proj, p1, p2,
                             matchup_df=matchup_df, savename=savename)

    # Make a table of suggested adds


def player_effects(db, week, p1, p2):

    team1 = db.manager_to_teamID(p1)
    team2 = db.manager_to_teamID(p2)

    

    res = {"Overall":[]}
    proj_dict = {}
    preds_dict = {}
    rosters = {}
    date_range = utils.date_range(db.week_date_range(week)[0], db.week_date_range(week)[1])
    effects = []
    for date in tqdm(date_range):

        # Base prediction -- as it actually was
        scores = db.matchup_score(week, date)
        proj = pred.proj_all_players(db, date)
        preds = pred.predict_matchup(db, date, team1, team2, proj=proj, scores=scores)
        proj_dict[date] = proj
        preds_dict[date] = preds

        # Now try replacing individual predictions with their actual performances for the day
        entries = []
        for it, team in enumerate([team1, team2]):
            contribs = db.player_contributions(team, date=date)
            if len(contribs) == 0:
                continue
            for name, performance in contribs.iterrows():
                new_scores = add_to_scores(scores, performance, team)
                # Modify proj to remove the game to play
                proj.at[(team, name), "exp_num_to_play"] += -1
                new_preds = pred.predict_matchup(db, date, team1, team2, proj=proj, scores=new_scores)
                proj.at[(team, name), "exp_num_to_play"] += 1
                entry = {}
                entry["name"] = name
                entry["date"] = date
                entry["manager"] = db.teamID_lookup(team)[0]
                entry["Overall"] = -(new_preds[0][1-it] - preds[0][1-it])
                for stat in new_preds[1]:
                    entry[stat] = -(new_preds[1][stat][0][1-it] - preds[1][stat][0][1-it])
                    if "%" not in stat:
                        entry[stat+"_quant"] = performance[stat] - proj.at[(team, name), stat]
                entries.append(entry)
        effects += entries

    return pd.DataFrame(effects)



# NOT WORKING YET
def streaming_effects(db, week, p1, p2):

    
    team1 = db.manager_to_teamID(p1)
    team2 = db.manager_to_teamID(p2)
    
    rosters = db.get_fantasy_rosters()
    rosters = rosters[rosters.teamID.isin([team1, team2])]

    date_range = utils.date_range(db.week_date_range(week)[0], db.week_date_range(week)[1])

    # Get rosters on last day of previous week
    rosters_prev = rosters[rosters.week == week-1]
    rosters_prev = rosters_prev[rosters_prev.date == rosters_prev.date.iloc[-1]]
    rosters_prev.set_index("teamID", inplace=True)
    proj_prev = pred.proj_all_players(db, db.week_date_range(week-1)[1])

    # Rosters for this week
    rosters = rosters[rosters.week == week]
    rosters.set_index(["teamID", "date"], inplace=True)

    # Get all the players who were dropped -- i.e. if you had
    # just kept the team the same
    # And the ones who were added
    dropped = {team1:{}, team2:{}}
    added = {team1:{}, team2:{}}
    for date in date_range:
        for team in [team1, team2]:
            rosters_today = rosters.loc[(team, date)]
            for name in rosters_prev.loc[team]["name"].values:
                if name not in rosters_today["name"] and name not in dropped[team]:
                    dropped[team][name] = date
            for name in rosters_today.name.values:
                if name not in rosters_prev.loc[team]["name"] and name not in added[team]:
                    added[team][name] = date
    
    
    # # Base prediction -- as it actually was
    # scores = db.matchup_score(week, date)
    # proj = pred.proj_all_players(db, date)
    # preds = pred.predict_matchup(db, date, team1, team2, proj=proj, scores=scores)

    stats_kept  = {team1:[], team2:[]}
    stats_steam = {team1:[], team2:[]}
    # for team in [team1, team2]:
    #     for player in stats_kept:


    for date in tqdm(date_range):
        for it, team in enumerate([team1, team2]):
            
            # Get the players that have been dropped
            contribs = db.player_contributions(team, date=date)
            if len(contribs) == 0:
                continue
            for name, performance in contribs.iterrows():
                new_scores = add_to_scores(scores, performance, team)
                # Modify proj to remove the game to play
                proj.at[(team, name), "exp_num_to_play"] += -1
                new_preds = pred.predict_matchup(db, date, team1, team2, proj=proj, scores=new_scores)
                proj.at[(team, name), "exp_num_to_play"] += 1
                entry = {}
                entry["name"] = name
                entry["date"] = date
                entry["manager"] = db.teamID_lookup(team)[0]
                entry["Overall"] = -(new_preds[0][1-it] - preds[0][1-it])
                for stat in new_preds[1]:
                    entry[stat] = -(new_preds[1][stat][0][1-it] - preds[1][stat][0][1-it])
                    if "%" not in stat:
                        entry[stat+"_quant"] = performance[stat] - proj.at[(team, name), stat]
                entries.append(entry)

    
    streams = {team1:{}, team2:{}}

    return

def add_to_scores(scores, performance, teamID):
    to_return = scores.copy(deep=True)
    for stat in utils.STATS_COLS:
        to_return.at[teamID, stat] += performance[stat]
    return to_return



if __name__ == "__main__":
    
    doPreds = False
    updateDB = False
    doSummaries = True
    date = "2024-02-17"
    week = 17
    db_file = "yahoo_fantasy_2023_24.sqlite"
    db = dbInterface(db_file)
    if updateDB:
        dbBuilder(db_file, debug=True).update_db()
    if doPreds:

        # retrospective = os.path.join('matchup results', '2022-2023', f'week{week}', 'retrospective.png')
        # past_preds(sc, gm, curLg, week, retrospective)

        predsSaveDir = os.path.join('matchup results', '2023-2024', f'week{week}', 'predictions')
        os.makedirs(predsSaveDir,exist_ok=True)
        run_predictions(db, week=week, folder=predsSaveDir, date=date)

    if doSummaries:
        scores = db.matchup_score(week-1)
        for matchup in scores.matchupNumber.unique():
            p1, p2 = scores[scores.matchupNumber==matchup]["manager"].values
            print("Doing Summary for", p1, "vs", p2)
            savedir = os.path.join('matchup results', '2023-2024', f'week{week-1}', 'summaries')
            os.makedirs(savedir,exist_ok=True)
            savename = os.path.join(savedir, f"{p1}_{p2}_summary.png")
            plot_matchup_history(db, week-1, p1, p2, savename=savename)

    # plot_matchup_history(db, 16, "Fabio", "Chi Yen")

    # ef = player_effects(db, 16, "Kayla", "Kai")