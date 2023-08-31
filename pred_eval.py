from tqdm import tqdm
from itertools import product
from pathlib import Path
import time
from scipy.stats import gaussian_kde, linregress, poisson, norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import utils
import pred
from db_interface import dbInterface


simple_stats = ["PTS", "3PTM", "REB", "AST", "ST", "BLK", "TO"]
percent_stats = {"FG%": ("FGA", "FGM"), "FT%": ("FTA", "FTM")}

def eval_wins(db, pred_mat, order, matchup_df, week):
    res = []
    for p1, p2 in product(matchup_df['teamID'].values, repeat=2):
        if p1 > p2:
            # matchup = grouped.get_group(i)
            # p1 = matchup['teamID'].iloc[0]
            # p2 = matchup['teamID'].iloc[1]
            matchup = matchup_df.loc[[p1, p2]]

            # Determine who won
            wins = [0,0,0]
            for stat in simple_stats+list(percent_stats.keys()):
                vals = matchup[stat].values
                tie = vals[0] == wins[1]
                if tie:
                    winner = 2
                else:
                    winner = np.argmax(vals)
                if stat == "TOV":
                    winner = 1-winner
                wins[winner]+=1
            tie = wins[0] == wins[1]
            if tie:
                winner = 2
            else:
                winner = np.argmax(wins)

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
    
    # Get every 10% chunk independent
    res["decade"] = (res["prob_win"]//0.1).astype(int)

    return res


def ratio_range(attempts, made, samples=10000, current_score=(0,0)):

    # Attempts as Poisson process
    a = np.random.poisson(attempts, samples) + current_score[0]
    m = np.random.poisson(made, samples) + current_score[1]
   
    r = m/a
    r[r > 1] = 1
    
    return np.mean(r), np.std(r)


def eval_stats(proj, matchup_df, week):

    res = []
    for teamID in proj:
        for stat in simple_stats + list(percent_stats.keys()):
            entry = {"teamID": teamID, "stat": stat, "week": week}
            x = matchup_df.at[teamID, stat]
            # Percent Stats
            if stat in percent_stats:
                entry["type"] = "percent"
                attempts, made = percent_stats[stat]
                mu, sigma = ratio_range(proj[teamID][attempts], proj[teamID][made])
                prob = norm.cdf(x, loc=mu, scale=sigma)
            # Counting stats
            else:
                entry["type"] = "counting"
                mu = proj[teamID][stat]
                prob = poisson.cdf(x, mu=mu)

            entry["pred"] = mu
            entry["act"] = x
            entry["prob"] = prob
            res.append(entry)


    return pd.DataFrame(res)


def plot_stat_kde(stats):

    f, ax = plt.subplots(nrows=3,ncols=3, figsize=(20,20))
    grouped = stats.groupby("stat")
    for i,stat in enumerate(grouped.indices):
        res = grouped.get_group(stat)

        row = int(i/3)
        col = i - int(i/3)*3

        kernel = gaussian_kde(res.prob.values)
        x = np.linspace(res.prob.min(), res.prob.max(), 100)
        y = kernel(x)

        ax[row, col].plot(x,y)
        ax[row, col].set_title(stat)
        if "%" in stat:
            ax[row, col].set_xlabel("Prob Density of Actual Result")
        else:
            ax[row, col].set_xlabel("Prob of Actual Result")
        ax[row,col].set_ylabel("KDE of Counts")
    
    plt.savefig("stat_kde.png")


def plot_stats_kde(stats_df_, stat, folder):

    stats_df = stats_df_[stats_df_["stat"] == stat]

    f, ax = plt.subplots(ncols=4, nrows=2)
    stats_by_day = stats_df.groupby("days_left_in_week")
    for n in range(7,0,-1):

        # KDE of cumulative probabilities
        stats = stats_by_day.get_group(n)
        kernel = gaussian_kde(stats.prob.values, bw_method=0.05)
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
        ax[i,j].set_xlabel("Probability of Getting <= Final Value")
        ax[i,j].set_ylabel("Fraction of Instances")
        ax[i,j].set_ylim((0,0.05))
        ax[i,j].set_xlim((0,1))

    f.set_size_inches((14,7))
    plt.suptitle(f"{stat}")
    plt.tight_layout()
    plt.savefig(Path(folder, f"{stat}_kde.png"))
    plt.close()

    return



def plot_win_scatter(wins_df, title="", savename="win_scatter.png"):

    f, ax = plt.subplots(ncols=4, nrows=2)
    wins_by_day = wins_df.groupby("days_left_in_week")
    for n in range(7,0,-1):
        wins = wins_by_day.get_group(n)
        grouped = wins.groupby("decade")
        x = []
        y = []
        for dec in grouped.groups:
            entries = grouped.get_group(dec)
            expected = np.mean(entries["prob_win"])
            actual = np.sum(entries["won"])/entries.shape[0]
            # print(dec, entries.shape[0])
            x.append(expected)
            y.append(actual)

        slope, intercept, r, p, se = linregress(x, y)
        i = 1 if n < 4 else 0
        j = 3 - n%4
        ax[i,j].grid()
        ax[i,j].scatter(x, y)
        ax[i,j].set_title(f"{n} Days Left in Week \n (slope: {np.round(slope, 2)}, intercept: {np.round(intercept, 2)}, r: {np.round(r**2,2)})")
        ax[i,j].set_xlabel("Expected Win Rate")
        ax[i,j].set_ylabel("Actual Win Rate")
        ax[i,j].set_ylim((0,1))
        ax[i,j].set_xlim((0,1))

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
    end_day = db.week_date_range(endWk)[1]
    wins = []
    stats = []
    days_left = 0
    past_week = 0
    for date in tqdm(utils.date_range(begin_day, end_day)[::-1]):

        week = db.week_for_date(date)
        if week != past_week:
            past_week = week
            days_left = 1
        else:
            days_left += 1
        
        # Ignore the beginning of the long all star week
        if days_left > 7:
            continue
        
        scoreboard = db.matchup_score(week, date)
        pred_mat, order, stat = pred.matchup_matrix(db, date, actual_played=True)
    
        d1 = eval_stats(stat, scoreboard, week)
        d2 = eval_wins(db, pred_mat, order, scoreboard, week)

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

    wins.to_csv("wins.csv", index=False)
    stats.to_csv("stats.csv", index=False)

    wins = pd.read_csv("wins.csv")
    stats = pd.read_csv("stats.csv")

    wins["decade"] = (wins["prob_win"]//0.1).astype(int)
    plot_win_scatter(wins, title="Predictions from Morning of n-th Day in Week",
                    savename="pred_eval_figs/win_scatter.png")
    
    for stat in simple_stats + list(percent_stats.keys()):
        plot_stats_kde(stats, stat, folder="pred_eval_figs")
        
    # stats.to_csv("stats.csv")
