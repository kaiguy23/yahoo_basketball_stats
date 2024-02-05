from pathlib import Path
import os

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

def plot_matchup_history():

    res = {"Overall":[]}
    # week = 17
    # team1, team2 = "418.l.20454.t.8", "418.l.20454.t.7"
    # week = 22
    # team1, team2 = "418.l.20454.t.8", "418.l.20454.t.4"
    # week = 11
    # team1, team2 = "418.l.20454.t.8", "418.l.20454.t.10"
    # week = 20
    # team1, team2 = "418.l.20454.t.8", "418.l.20454.t.2"
    week = 21
    team1, team2 = "418.l.20454.t.8", "418.l.20454.t.11"
    date_range = utils.date_range(db.week_date_range(week)[0], db.week_date_range(week)[1])
    for date in date_range:
        #  res.append(predict_matchup(db, date, "418.l.20454.t.8", "418.l.20454.t.4")[0])
        preds = predict_matchup(db, date, team1, team2)
        res["Overall"].append(preds[0])
        for stat in preds[1]:
            if stat not in res:
                res[stat] = []
            res[stat].append(preds[1][stat])
    
    final_scoreboard = db.matchup_score(week)
    for stat in res:
        res[stat] = np.vstack(res[stat])
        if stat == "Overall":
            winners = utils.matchup_winner(final_scoreboard.loc[team1], final_scoreboard.loc[team2])
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
    
    
    xlabels = date_range + ["Final"]
    fig, axd = plt.subplot_mosaic([["Overall", "Overall", "Overall"],
                                   ["FG%", "FT%", "3PTM"],
                                   ["PTS", "REB", "AST"],
                                   ["ST", "BLK", "TO"]],
                              figsize=(9, 12), layout="constrained")
    
    fig.suptitle(f"Predicted Results from Morning of Specified Day (Week {week})")
    for stat in res:
        # for k in axd:
        #     annotate_axes(axd[k], f'axd["{k}"]', fontsize=14)
        ax = axd[stat]
        ax.plot(res[stat][:,0], label=db.teamID_lookup(team1)[0], marker=".", lw=1)
        ax.plot(res[stat][:,1], label=db.teamID_lookup(team2)[0], marker=".", lw=1)
        ax.plot(res[stat][:,2], label="Tie", marker=".", lw=1)
        ax.set_title(stat)
        ax.grid()
        if stat == "Overall":
            ax.set_ylabel("Probability")
            ax.set_xlabel("Date")
            ax.legend()
            ax.set_xticks(ticks=np.arange(len(xlabels)), labels=xlabels)
        else:
            ax.set_xticks(ticks=np.arange(len(xlabels)), labels=[""]*len(xlabels))
    # plt.tight_layout()
    plt.savefig("test.png")


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
                mu = stat_victory[stat][ip+1][0]
                sigma = stat_victory[stat][ip+1][1]
                x = (norm.ppf(epsilon, loc=mu, scale=sigma),
                    norm.ppf(1-epsilon,loc=mu, scale=sigma))
                # prob = norm.pmf(x, loc=mu, scale=sigma)
            else:
                mu = stat_victory[stat][ip+1][0]
                x = np.arange(poisson.ppf(epsilon, mu),
                    poisson.ppf(1-epsilon, mu)+1)
                # prob = poisson.pmf(x, mu)
            row = int(i/3)
            col = i-int(i/3)*3
            a = ax[row,col]

            if ip == 0:
                vic = np.array([np.round(x*100,2) for x in stat_victory[stat][0]])
                a.set_title(f"{stat} - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%")

            a.errorbar(mu, (1-ip)*0.01, xerr=np.array((mu-x[0], x[-1]-mu)).reshape(2,1),
                       label=p, capsize = 4, fmt = 'o', alpha=0.75)
            a.set_ylim([-0.005,0.015])
            a.set_yticks([])
            if ip == 1:
                a.legend()
    
    if savename is None:
        savename = f"{p1}_vs_{p2}.png"
    plt.savefig(savename)

    return


def run_predictions(db: dbInterface, week: int, folder: str, today: bool = False):
    """
    Does the weekly prediction/figure generation
    for the next week.

    Args:
        db (dbInterface): dbInterface object
        week (int): fantasy week to do predictions for
        folder (str): folder to save figures to
    """

    d0, df = db.week_date_range(week)
    if today:
        d0 = max(d0, utils.TODAY_STR)
    print("Doing Predictions for Morning of", d0)
    proj = pred.proj_all_players(db, d0)
    pred_mat, order, stats = pred.matchup_matrix(db, d0, actual_played=False)
    
    matchup_df = db.matchup_score(week, d0)

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
        plot_matchup_summary(db, d0, proj, p1, p2,
                             matchup_df=matchup_df, savename=savename)

    # Make a table of suggested adds


def run_past_predictions(db, week):
    return


if __name__ == "__main__":
    
    doPreds = True
    updateDB = True
    today = False
    week = 16
    db_file = "yahoo_fantasy_2023_24.sqlite"
    db = dbInterface(db_file)
    if updateDB:
        dbBuilder(db_file, debug=True).update_db()
    if doPreds:

        # retrospective = os.path.join('matchup results', '2022-2023', f'week{week}', 'retrospective.png')
        # past_preds(sc, gm, curLg, week, retrospective)

        predsSaveDir = os.path.join('matchup results', '2023-2024', f'week{week}', 'predictions')
        os.makedirs(predsSaveDir,exist_ok=True)
        run_predictions(db, week=week, folder=predsSaveDir, today=today)
