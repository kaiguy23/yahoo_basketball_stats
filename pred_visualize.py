import numpy as np
import pandas as pd
import seaborn as sn

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pred


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
            i,j = matchup_df[matchup_df['matchupNumber']==m].index
            ax.add_patch(Rectangle((i,j), 1, 1, fill=False, edgecolor='blue', lw=3))
            ax.add_patch(Rectangle((j,i), 1, 1, fill=False, edgecolor='blue', lw=3))

    if matchup_df is None:
            f.suptitle(f"NBA Fantasy Predicted Results", fontsize = 30)
    else:
        f.suptitle(f"NBA Fantasy Predicted Results (Week {matchup_df.iloc[0].week})", fontsize = 30)
   
    if savename != "":
        plt.savefig(savename)
        plt.close(f)


def plot_matchup_summary(proj, p1, p2, matchup_df = None, savename=None):
    """
    Plots a summary of the specified matchup, showing the 90% confidence interval
    for each stat, showing the probability that either player will 
    win the category.


    Args:
        proj (dict): dictionary that maps player to projected stats
        p1 (str): name of the first player
        p2 (str): name of the second player
        matchup_df (pandas df, optional): matchup df with the current score for midweek analysis. Defaults to None.
    """

    p,s,m = prob_victory(proj, p1, p2, matchup_df=matchup_df)
    
    f, ax = plt.subplots(nrows=3,ncols=3, figsize=(20,14))

    vic = [np.round(x*100,2) for x in p]
    f.suptitle(f"Probability of Victory - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%", fontsize=26)

    # epsilon = 0.0001
    epsilon = 0.05

    if not matchup_df is None:
        grouped = matchup_df.groupby("manager")

    
   
    for ip, p in enumerate((p1,p2)):
        for i, stat in enumerate(s):
            if "%" in stat:
                mu = m[stat][p][0]
                sigma = m[stat][p][1]
                x = (norm.ppf(epsilon, loc=mu, scale=sigma),
                    norm.ppf(1-epsilon,loc=mu, scale=sigma))
                # prob = norm.pmf(x, loc=mu, scale=sigma)
            else:
                mu = proj[p][stat]
                x = np.arange(poisson.ppf(epsilon, mu),
                    poisson.ppf(1-epsilon, mu)+1)
                # prob = poisson.pmf(x, mu)
            row = int(i/3)
            col = i-int(i/3)*3
            a = ax[row,col]

            if ip == 0:
                vic = np.array([np.round(x*100,2) for x in s[stat]])
                a.set_title(f"{stat} - {p1}: {vic[0]}%, {p2}: {vic[1]}%, Tie: {vic[2]}%")
            # Add current stats
            exp_val = mu
            if "%" not in stat and not matchup_df is None:
                exp_val+=grouped.get_group(p)[stat].iloc[0]
            a.errorbar(exp_val, (1-ip)*0.01, xerr=np.array((mu-x[0], x[-1]-mu)).reshape(2,1), label=p, capsize = 4, fmt = 'o', alpha=0.75)
            a.set_ylim([-0.005,0.015])
            a.set_yticks([])
            if ip == 1:
                a.legend()
    
    if savename is None:
        savename = f"{p1}_vs_{p2}.png"
    plt.savefig(savename)

    return
