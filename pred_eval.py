from tqdm import tqdm
from itertools import product
from pathlib import Path
import time
from scipy.stats import gaussian_kde, linregress, poisson, norm
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import utils
import pred
from db_interface import dbInterface


simple_stats = ["PTS", "3PTM", "REB", "AST", "ST", "BLK", "TO"]
percent_stats = {"FG%": ("FGA", "FGM"), "FT%": ("FTA", "FTM")}

def eval_wins(db, pred_mat, order, final_scores, week):
    res = []
    for p1, p2 in product(final_scores['teamID'].values, repeat=2):
        if p1 > p2:
            matchup = final_scores.loc[[p1, p2]]

            # Determine who won
            wins, winner = utils.matchup_winner(matchup.iloc[0],
                                                matchup.iloc[1])
            i1 = order.index(db.teamID_lookup(p1)[0])
            i2 = order.index(db.teamID_lookup(p2)[0])
            entry1 = {"teamID": p1, "opponent": p2,
                    #   "matchup_number": i,
                    "score": tuple(wins),
                    "winning_team": [p1, p2, "tie"][winner],
                    "won": winner == 0,
                    "tied": winner == 2, 
                    "prob_win": pred_mat[i1, i2],
                    "prob_tie": 1 - pred_mat[i1, i2] - pred_mat[i2, i1]}
            entry2 = {"teamID": p2, "opponent": p1,
                    #   "matchup_number": i,
                    "score": (wins[1], wins[0], wins[2]),
                    "winning_team": [p1, p2, "tie"][winner],
                    "won": winner == 1,
                    "tied": winner == 2,
                    "prob_win": pred_mat[i2, i1],
                    "prob_tie": 1 - pred_mat[i1, i2] - pred_mat[i2, i1]}
            
            res.append(entry1)
            res.append(entry2)
        
    res =  pd.DataFrame(res)
    
    # Get every 20% chunk independent
    res["perc_group"] = (res["prob_win"]//0.2).astype(int)

    return res


def ratio_range(attempts, made, samples=10000, current_score=(0,0)):

    # Attempts as Poisson process
    if attempts > 0:
        a = np.random.poisson(attempts, samples) + current_score[0]
    else:
        a = np.zeros(samples) + current_score[0]
   
    # Made as Poisson process, each centered on the expected percentage
    # Big array of attempts and made
    m = np.zeros_like(a)
    for i in range(samples):
        if attempts > 0:
            exp = (a[i] - current_score[0])*(made/attempts)
            m[i] = np.random.poisson(exp) + current_score[1]
        else:
            m[i] = current_score[1]

    # Ratios i.e. percentage
    r = m/a
    r[r > 1] = 1
    
    return np.mean(r), np.std(r)


def eval_stats(proj, scoreboard, final_scores, week):

    res = []
    for teamID in sorted(list(proj.keys())):
        for stat in simple_stats + list(percent_stats.keys()):
            entry = {"teamID": teamID, "stat": stat, "week": week}

            # Current and final stats
            x = scoreboard.at[teamID, stat]
            xf = final_scores.at[teamID, stat]

            # Percent Stats
            if stat in percent_stats:
                entry["type"] = "percent"
                attempts, made = percent_stats[stat]
                mu, sigma = ratio_range(proj[teamID][attempts][0], proj[teamID][made][0],
                                        current_score=(proj[teamID][attempts][1],
                                                       proj[teamID][made][1]))
                prob = norm.cdf(xf, loc=mu, scale=sigma)
                if proj[teamID][attempts][1] > 0:
                    entry["already"] = proj[teamID][made][1]/proj[teamID][attempts][1]
                else:
                    entry["already"] = 0
                entry["pred"] = mu
                entry["pred_total"] = mu
            # Counting stats
            else:
                entry["type"] = "counting"
                # Predicted amount of stat to be gained
                mu = proj[teamID][stat][0]
                prob = poisson.cdf(xf-x, mu=mu)
                entry["already"] = x
                entry["pred"] = mu
                entry["pred_total"] = mu + x

            entry["final"] = xf
            entry["prob"] = prob
            res.append(entry)


    return pd.DataFrame(res)



def plot_stats_kde(stats_df_, stat, folder):

    stats_df = stats_df_[stats_df_["stat"] == stat]

    f, ax = plt.subplots(ncols=4, nrows=2)
    stats_by_day = stats_df.groupby("days_left_in_week")
    for n in range(7,0,-1):

        # KDE of cumulative probabilities
        stats = stats_by_day.get_group(n)
        kernel = gaussian_kde(stats.prob.values)
        n_samples = 100
        x = np.linspace(0, 1, n_samples)
        y = kernel(x)

        # Normalize y
        y = y/np.sum(y)


        i = 1 if n < 4 else 0
        j = 3 - n%4
        ax[i,j].grid()
        ax[i,j].plot(x, y)
        ax[i,j].set_title(f"{n} Days Left in Week")
        ax[i,j].set_xlabel("Point in CDF")
        ax[i,j].set_ylabel("Fraction of Instances")
        ax[i,j].set_ylim((0,0.05))
        ax[i,j].set_xlim((0,1))

    f.set_size_inches((14,7))
    plt.suptitle(f"{stat}")
    plt.tight_layout()
    plt.savefig(Path(folder, f"{stat}_kde.png"))
    plt.close()

    return


def plot_stats_kde_err(stats_df_, stat, folder):

    stats_df = stats_df_[stats_df_["stat"] == stat]

    lim = max(np.abs((stats_df.pred_total.values-stats_df.final.values).min()),
            (stats_df.pred_total.values-stats_df.final.values).max())

    f, ax = plt.subplots(ncols=4, nrows=2)
    stats_by_day = stats_df.groupby("days_left_in_week")
    for n in range(7,0,-1):

        # KDE of cumulative probabilities
        stats = stats_by_day.get_group(n)
        err = stats.pred_total.values-stats.final.values
        kernel = gaussian_kde(err)
        n_samples = 100
        # x = np.linspace(0, 1, n_samples)
        x = np.linspace(-lim, lim, n_samples)
        y = kernel(x)

        # Normalize y
        y = y/np.sum(y)

        # Calculate moments
        avg = np.sum(x*y)
        xsq = np.sum((x**2)*y)
        std = np.sqrt(xsq - avg**2)

        i = 1 if n < 4 else 0
        j = 3 - n%4
        ax[i,j].grid()
        ax[i,j].plot(x, y)
        if r"%" not in stat:
            ax[i,j].set_title(f"{n} Days Left in Week \n"+r"($\mu$ = " + str(np.round(avg, 1))\
                          +r" $\sigma$ = " + str(np.round(std, 1))+ ")")
        else:
            ax[i,j].set_title(f"{n} Days Left in Week \n"+r"($\mu$ = " + str(np.round(avg, 3))\
                          +r", $\sigma$ = " + str(np.round(std, 3))+ ")")
        ax[i,j].set_xlabel("Difference From Actual")
        ax[i,j].set_ylabel("Fraction of Instances")
        # ax[i,j].set_ylim((0,0.05))
        ax[i,j].set_xlim((-lim, lim))

    f.set_size_inches((14,7))
    plt.suptitle(f"{stat}")
    plt.tight_layout()
    plt.savefig(Path(folder, f"err_{stat}_kde.png"))
    plt.close()

    return

def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

def plot_win_scatter(wins_df, title="", savename="win_scatter.png",
                     bin_size=0.1, nbins=10):
    f, ax = plt.subplots(ncols=4, nrows=2)
    wins_by_day = wins_df.groupby("days_left_in_week")
    for n in range(7,0,-1):
        wins = wins_by_day.get_group(n)
        x = []
        y = []
        yerr = []
        # for dec in grouped.groups:
        for bin_center in np.linspace(bin_size,1-bin_size,nbins):
            entries = wins[np.logical_and(wins["prob_win"] > bin_center - bin_size,
                                          wins["prob_win"] < bin_center + bin_size)]
            expected = np.mean(entries["prob_win"])
            actual = np.mean(entries["won"].astype(int))
            x.append(expected)
            y.append(actual)
            std = np.std(entries["won"].astype(int))
            std_err = std/np.sqrt(entries.shape[0])
            yerr.append(std_err)
        
        a_fit,cov=curve_fit(linearFunc,x,y,sigma=yerr,absolute_sigma=True)
        inter = a_fit[0]
        slope = a_fit[1]
        d_inter = np.sqrt(cov[0][0])
        d_slope = np.sqrt(cov[1][1])

        i = 1 if n < 4 else 0
        j = 3 - n%4
        ax[i,j].grid()
        ax[i,j].errorbar(x, y, yerr=yerr, ls="", marker=".", markersize=10)
        ax[i,j].set_title(f"{n} Days Left in Week \n slope: {np.round(slope, 2)} += {np.round(d_slope, 3)}, "\
                          +f"\nintercept: {np.round(inter, 2)} += {np.round(d_inter, 3)}")
        ax[i,j].set_xlabel("Expected Win Rate")
        ax[i,j].set_ylabel("Actual Win Rate")
        ax[i,j].set_ylim((0,1))
        ax[i,j].set_xlim((0,1))
    
    # Plot % right calls as a function of days left in week
    days_left = np.arange(1, 8)
    right_call = np.zeros_like(days_left, dtype=float)
    ax[1, 3].grid()
    for i, n in enumerate(days_left):
        wins = wins_by_day.get_group(n)
        pred_win = wins["prob_win"] > 1 - wins["prob_win"] - wins["prob_tie"]
        act_win = wins["won"]
        right_call[i] = np.sum(np.logical_not(np.logical_xor(pred_win, act_win)))/wins.shape[0]

    ax[1, 3].plot(days_left, right_call, ls="", marker=".", markersize=15)
    ax[1, 3].set_title("Overall Prediction Accuracy")
    ax[1, 3].set_xlabel("Number of Days Left in Week")
    ax[1, 3].set_ylabel("'Right Call' Rate")
    ax[1, 3].set_ylim((0, 1))
    
    # Plot stuff about ties
    # days_left = np.arange(1, 8)
    # pred_tie = np.zeros_like(days_left, dtype=float)
    # act_tie = wins["tied"].sum()/wins.shape[0]
    # ax[1, 3].grid()
    # for i, n in enumerate(days_left):
    #     wins = wins_by_day.get_group(n)
    #     pred_tie[i] = wins["prob_tie"].mean()
        
    # ax[1, 3].plot(days_left, pred_tie, ls="", marker=".", markersize=10)
    # ax[1, 3].plot(days_left, np.ones_like(days_left)*act_tie, ls="--", c="k", alpha=0.5,
    #               label="Actual Tie Rate")
    # ax[1, 3].legend()
    # ax[1, 3].set_title("Ties")
    # ax[1, 3].set_xlabel("Number of Days Left in Week")
    # ax[1, 3].set_ylabel("Expected Tie Rate")
    # ax[1, 3].set_ylim((act_tie - 0.01, act_tie + 0.01))

    f.set_size_inches((14,7))
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()



if __name__ == "__main__":

    db = dbInterface("past_season_dbs/yahoo_fantasy_2022_23.sqlite")

    startWk = 6
    endWk = 19
    begin_day = db.week_date_range(startWk)[0]
    end_day = db.week_date_range(endWk)[1] ##TODO: CHange back to 1
    wins = []
    stats = []
    days_left = 0
    past_week = -1
    final_score = None
    for date in tqdm(utils.date_range(begin_day, end_day)[::-1]):

        week = db.week_for_date(date)
        if week != past_week:
            past_week = week
            days_left = 1
            final_score = db.matchup_score(week)
        else:
            days_left += 1
        
        # Ignore the beginning of the long all star week
        if days_left > 7:
            continue

        scoreboard = db.matchup_score(week, date)
        pred_mat, order, stat = pred.matchup_matrix(db, date, actual_played=True)
        # pred_mat, order, stat = pred.matchup_matrix(db, date, actual_played=False)
    
        d1 = eval_stats(stat, scoreboard, final_score, week)
        d2 = eval_wins(db, pred_mat, order, final_score, week)

        d2["days_left_in_week"] = days_left
        d2["week"] = week
        
        d1["days_left_in_week"] = days_left
        d1["week"] = week

        stats.append(d1)
        wins.append(d2)

    wins = pd.concat(wins)
    stats = pd.concat(stats)

    # wins.to_csv("wins_no_hindsight.csv")
    # wins = pd.read_csv("wins_no_hindsight.csv")
    # stats.to_csv("stats_no_hindsight.csv", index=False)
    # stats = pd.read_csv("stats.csv")

    wins.to_csv("wins.csv", index=False)
    wins = pd.read_csv("wins.csv")
    stats.to_csv("stats.csv", index=False)
    stats = pd.read_csv("stats.csv")

    stats = stats[np.logical_not(np.isnan(stats.prob.values))]

    plot_win_scatter(wins, title="Predictions from Morning of n-th Day in Week",
                    savename="pred_eval_figs/win_scatter.png", nbins=5,
                    bin_size=0.1)
    # plot_win_scatter(wins, title="Predictions from Morning of n-th Day in Week",
    #                 savename="pred_eval_figs/win_scatter_no_hindsight.png", nbins=10,
    #                 bin_size=0.1)
    
    for stat in simple_stats + list(percent_stats.keys()):
        plot_stats_kde_err(stats, stat, folder="pred_eval_figs")
    for stat in simple_stats + list(percent_stats.keys()):
        plot_stats_kde(stats, stat, folder="pred_eval_figs")
        