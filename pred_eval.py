from pred import *
from tqdm import tqdm
import time
from scipy.stats import gaussian_kde

simple_stats = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV"]
percent_stats = {"FG%": ("FGA", "FGM"), "FT%": ("FTA", "FTM")}

def eval_wins(proj, matchup_df, week):
    res = []
    grouped = matchup_df.groupby("matchupNumber")
    for i in grouped.indices:
        matchup = grouped.get_group(i)
        p1 = matchup['manager'].iloc[0]
        p2 = matchup['manager'].iloc[1]

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

        p, s, m = prob_victory(proj, p1, p2)

        entry = {"p1": p1, "p2": p2, "score": wins,
                "winner": [p1, p2, "tie"][winner],
                "prob": p[winner]}
        res.append(entry)
        
    return pd.DataFrame(res)


def eval_stats(proj, matchup_df, week):

    grouped = matchup_df.groupby("manager")
    res = []
    for p in proj:
        
        for i, stat in enumerate(simple_stats):
            entry = {"manager": p, "stat": stat, "week": week, "type": "counting"}
            mu = proj[p][stat]
            x = grouped.get_group(p)[stat].iloc[0]
            prob = poisson.pmf(x, mu)
            entry["pred"] = mu
            entry["act"] = x
            entry["prob"] = prob
            res.append(entry)

        # Percent Stats
        for stat in percent_stats:
            entry = {"manager": p, "stat": stat, "week": week, "type": "percent"}
            attempts, made = percent_stats[stat]
            mu, sigma = ratio_range(proj[p][attempts], proj[p][made])
            x = grouped.get_group(p)[stat].iloc[0]
            prob = norm.pdf(x, loc=mu, scale=sigma)
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


def plot_win_kde(wins):

    f, ax = plt.subplots()
    kernel = gaussian_kde(wins.prob.values)
    x = np.linspace(wins.prob.min(), wins.prob.max(), 100)
    y = kernel(x)
    plt.plot(x,y)
    plt.xlabel("Probability of Victory")
    plt.ylabel("KDE of counts")
    plt.savefig("wins_kde.png")



if __name__ == "__main__":

    sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')

    curWk = curLg.current_week()
    wins = []
    stats = []
    for week in tqdm(range(6,curWk)):
        proj, matchup_df = past_preds(week, sc, gm, curLg)
        d1 = eval_stats(proj, matchup_df, week)
        d2 = eval_wins(proj, matchup_df, week)
        stats.append(d1)
        wins.append(d2)
        time.sleep(60*60)
        sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')
    wins = pd.concat(wins)
    stats = pd.concat(stats)

    wins.to_csv("wins.csv")
    stats.to_csv("stats.csv")
