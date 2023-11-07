import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
from scipy.stats import skellam, poisson, norm
import pandas as pd
import copy
from pathlib import Path
from ast import literal_eval
import os

from itertools import product

plt.switch_backend("Agg")

from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa


from db_interface import dbInterface
from db_builder import dbBuilder
import utils



CORE_STATS = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
       'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK','PTS','NBA_FANTASY_PTS']


GKERN_SIG = 3


def gkern_1sided(size: int, sig: float = GKERN_SIG) -> np.array:
    """
    Creates a one sided gaussian kernel of length size,
    with peak at the 0 index, and std of sigma.

    Kernel is normalized to sum to 1.

    Args:
        size (int): length of kernel to produce
        sig (float, optional): Sigma (std) for Gaussian. Defaults to GKERN_SIG.

    Returns:
        np.array[float]: Array of length size with value 1 at index 0,
                         and a Gaussian falloff with std sig.
    """
    x = np.arange(size)
    gauss = np.exp(-0.5 * (x**2) / (sig**2))
    return gauss/np.sum(gauss)


def skellam_prob(mu1: float, mu2: float,
                 current_score: tuple[int] = (0,0),
                 epsilon = 1e-05) -> tuple[float]:
    """
    Calculates the probability of mu1 winning
    or mu2 winning or it being a tie

    Args:
        mu1 (float): mean of first Poisson distribution
        mu2 (float): mean of second Poisson distribution
        current_score (tuple[int]): current scores
        epsilon (float): probability below which to ignore

    returns:
        tuple (probability dist 1 is higher, probability dist 2 is higher, prob of equal values)
    """
    if mu1 > 0 and mu2 > 0:
        # Probability of final score differences 
        # player1 - player2
        x = np.arange(skellam.ppf(epsilon, mu1, mu2) ,
                        skellam.ppf(1-epsilon, mu1, mu2)+1)
        prob = skellam.pmf(x, mu1, mu2)

        # Adjust x to reflect current score
        x += current_score[0]-current_score[1]
    # zero mu -- no games left
    elif mu2 == 0:
        # Amount player 1 might score
        x = np.arange(poisson.ppf(epsilon, mu1) ,
                        poisson.ppf(1-epsilon, mu1)+1)
        prob = poisson.pmf(x, mu1)
        # Adjust x to reflect current score
        x += current_score[0]-current_score[1]

    else:
        # Amount player 2 might score
        # minus to remain in the p1-p2 paradigm
        x = -np.arange(poisson.ppf(epsilon, mu2) ,
                        poisson.ppf(1-epsilon, mu2)+1)
        prob = poisson.pmf(-x, mu2)
        # Adjust x to reflect current score
        x += current_score[0]-current_score[1]

    # dist 2 > dist 1
    w2 = np.sum(prob[x < 0])

    # tie
    tie = np.sum(prob[x==0])

    # dist 1 > dist 2
    w1 = np.sum(prob[x > 0])

    # super high prob correction
    # if 1 - w1 < epsilon:
    #     w2 = epsilon/2
    #     tie = epsilon/2
    # if 1 - w2 < epsilon:
    #     w1 = epsilon/2
    #     tie = epsilon/2
    norm_factor = w2 + tie + w1

    return (w1/norm_factor, w2/norm_factor, tie/norm_factor)


def ratio_prob(attempts: tuple[int], made: tuple[int],
               samples:int = 10000,
               current_score: tuple[tuple[int]] = ((0,0), (0,0))) -> (tuple[float],
                                                                      tuple[float],
                                                                      tuple[float]):
    """
    Randomly samples attempts as a poisson distribution samples times, and then samples
    made as another poisson distribution centered on the expected fg percent.


    Args:
        attempts (tuple): number of attempts (p1, p2)
        made (tuple): number made (p1, p2)
        samples (int, optional): number of samples for estimating distribution. Defaults to 10000.
        current_score (tuple): current score ((p1 attempts, p2 attempts), (p1 made, p2 made))

    Returns:
        three tuples, (prob p1 victory, prob p2 victory, prob tie), (p1 mean, p1 std), (p2 mean, p2 std)
    """
    # Random generator
    rng = np.random.default_rng()

    # Attempts as Poisson process
    if attempts[0] > 0:
        a1 = rng.poisson(attempts[0], samples) + current_score[0][0]
    else:
        a1 = np.zeros(samples) + current_score[0][0]
    if attempts[1] > 0:
        a2 = rng.poisson(attempts[1], samples) + current_score[0][1]
    else:
        a2 = np.zeros(samples) + current_score[0][1]

    a1[a1 == 0] = 1
    a2[a2 == 0] = 1

    # Made as Poisson process, each centered on the expected percentage
    # Big array of attempts and made
    m1 = np.zeros_like(a1)
    m2 = np.zeros_like(a2)
    for i in range(samples):
        if attempts[0] > 0:
            exp1 = (a1[i] - current_score[0][0])*(made[0]/attempts[0])
            m1[i] = rng.poisson(exp1) + current_score[1][0]
        else:
            m1[i] = current_score[1][0]
        if attempts[1] > 0:
            exp2 = (a2[i] - current_score[0][1])*(made[1]/attempts[1])
            m2[i] = rng.poisson(exp2) + current_score[1][1]
        else:
            m2[i] = current_score[1][1]
    
    # Ratios i.e. percentage
    r1 = m1/a1
    r1[r1 > 1] = 1
    
    r2 = m2/a2
    r2[r2 > 1] = 1
    
    comp = r1 - r2


    w1 = np.sum(comp > 0)/samples
    tie = np.sum(comp == 0)/samples
    w2 = np.sum(comp < 0)/samples
    
    return (w1, w2, tie), (np.mean(m1), np.std(m1)), (np.mean(m2), np.std(m2))


def proj_player_stats(db: dbInterface, name: str,
                      date: str = utils.TODAY_STR,
                      kern_sig: float = GKERN_SIG) -> dict[float]:
    """
    Projects player stats by taking a weighted average by a Gaussian
    kernel of std kern_sig of games played up to (but not including)
    the specified date.

    Args:
        db (dbInterface): db interface object
        name (str): name of player
        date (str, optional): Date in form YYYY-MM-DD.
                              Defaults to utils.TODAY_STR.
        kern_sig (float, optional): STD of Gaussian kernel. Defaults to GKERN_SIG.
        actual_layed (bool, optional): If reviewing past predictions, adjust the games
                                       played estimate to reflect the actual number of games
                                       played for each player.

    Returns:
        dict[float]: Maps statistic to projected value
    """

    # Get player stats up to the specified day
    stats = db.player_stats(name)
    stats = stats[stats["GAME_DATE"] < date]

    proj = {}
    # Reverse it because most recent games are last
    kern = gkern_1sided(stats.shape[0], kern_sig)[::-1]
    for cat in utils.STATS_COLS + ['NBA_FANTASY_PTS']:
        if stats.shape[0] > 0:
            proj[cat] = np.mean(stats[cat].values)
            # proj[cat] = np.sum(kern*stats[cat].values)
        else:
            proj[cat] = 0
    return proj


def proj_all_players(db: dbInterface, date: str = utils.TODAY_STR,
                     kern_sig: float = GKERN_SIG,
                     teamIDs: list[str] = [],
                     actual_played: bool = False) -> pd.DataFrame:
    """
    Returns a dataframe with projected stats for all rostered
    players on the morning of the specified 
    date (i.e., before any games have been played).

    Additionally says how many games they've played and remain to
    be played over the course of the week in question.

    Args:
        db (dbInterface): db Interface object
        date (str, optional): Date in form YYYY-MM-DD.
                              Defaults to utils.TODAY_STR.
        kern_sig (float, optional): STD of Gaussian kernal. Defaults to GKERN_SIG.

    Returns:
        pd.DataFrame: Dataframe with player names, projected stats,
                      and their fantasy status.
    """

    week = db.week_for_date(date)
    rosters = db.fantasy_rosters(date)
    if teamIDs:
        rosters = rosters[rosters['teamID'].isin(teamIDs)]
    df = []
    
    # Rostered players
    for i, row in rosters.iterrows():
        # Copy over info
        entry = {}
        entry["name"] = row["name"]
        entry.update(proj_player_stats(db, row["name"], date, kern_sig))
        to_copy = ["selected_position", "status", "manager", "teamName", "teamID",
                   "eligible_positions"]
        for col in to_copy:
            entry[col] = row[col]

        # Add games played
        nba_team = row["nba_team"]
        if nba_team != "N/A":
            games_played = db.games_in_week(nba_team, week, upto=date)
            entry["games_played"] = games_played[0]
            entry["games_to_come"] = games_played[2]
        else:
            entry["games_played"] = 0
            entry["games_to_come"] = 0

        entry["nba_team"] = nba_team
        
        df.append(entry)

    df = pd.DataFrame(df)
    df.set_index(["teamID", "name"], inplace=True, drop=False)

    if actual_played:
        final_date = db.week_date_range(db.week_for_date(date))[1]
        df["exp_num_to_play"] = np.zeros(df.shape[0], dtype=int)
        return adjust_for_actual_played(db, df, date, final_date, kern_sig)
    else:
        calc_multiplier(df)
        return df


def calc_actual_played(db: dbInterface, teamID: str, start: str, end: str):
    """
    Looks back through nba game logs to determine how many games were 
    played by players for the specified team between the two specified
    dates (inclusive on both ends).

    Only counts if the player was in an active position in fantasy.

    Note: If a player didn't play, but wasn't on the injury report,
    it won't show up as a game on here, vs. it will on Yahoo.

    Args:
        db (dbInterface): dbInterface object
        teamID (str): teamID
        start (str): start date in YYYY-MM-DD format
        end (str): end date in YYYY-MM-DD format

    Returns:
        dict: maps player name to number played
    """
    
    sql_filter = f"WHERE teamID LIKE '{teamID}' "
    sql_filter += f"AND GAME_DATE >= '{start}' "
    sql_filter += f"AND GAME_DATE <= '{end}' "
    entries = db.get_nba_stats(sql_filter)

    # Filter out players who weren't in active positions
    entries = entries[entries["selected_position"].isin(utils.ACTIVE_POS)]

    counts = {}
    for i, row in entries.iterrows():
        if row["PLAYER_NAME"] not in counts:
            counts[row["PLAYER_NAME"]] = 1
        else:
            counts[row["PLAYER_NAME"]] += 1

    return counts


def adjust_for_actual_played(db: dbInterface, proj: pd.DataFrame,
                             start: str, end: str,
                             kern_sig: float = GKERN_SIG):

    for team in proj["teamID"].unique():
        counts = calc_actual_played(db, team, start, end)
        for player in counts:
            if (team, player) in proj.index:
                proj.at[(team, player), "exp_num_to_play"] = counts[player]
            else:
                new_entry = proj_player_stats(db, player, start, kern_sig)
                new_entry["name"] = player
                new_entry["teamID"] = team
                new_entry["exp_num_to_play"] = counts[player]
                new_entry["games_played"] = 0
                new_entry = pd.DataFrame([new_entry])
                new_entry.set_index(["teamID", "name"], inplace=True, drop=False)
                proj = pd.concat([proj, new_entry])
        
        # Go through and remove games played if the actual was 0 and
        # wouldn't show up in counts
        for player in proj.loc[team].name.values:
            if player not in counts:
                proj.at[(team, player), "exp_num_to_play"] = 0

    return proj



def calc_multiplier(proj: pd.DataFrame, injured = ["INJ", "NA", "O"]):
    """
    Calculates the expected number of games played and
    adds it on as a column to proj

    Currently just takes the top 10 players on any given day
    by expected fantasy point output

    Args:
        proj (pd.DataFrame): projections from proj_all_players
    """
    exp_to_play = {}
    for i in proj.index:
        exp_to_play[i] = 0
    for teamID in proj.index.levels[0]:
        team_roster = proj.loc[teamID]
        # Ignore players that are injured or on IL
        team_roster = team_roster[np.logical_not(team_roster["status"].isin(injured))]
        team_roster = team_roster[["IL" not in x for x in team_roster["selected_position"].values]]
        remaining_mat = np.vstack(team_roster["games_to_come"].values)
        for j in range(remaining_mat.shape[1]):
            subset = team_roster[remaining_mat[:,j].astype(bool)]
            if subset.shape[0] <= 10:
                for name in subset["name"].values:
                    exp_to_play[(teamID, name)] += 1
            else:
                subset = subset.sort_values("NBA_FANTASY_PTS", ascending=False)
                for i in range(10):
                    exp_to_play[(teamID, subset.iloc[i]["name"])] += 1
                    

            # assigned = assign_roster_spots(subset)
            # for name in assigned["ACTIVE"]:
            #     exp_to_play[(teamID, name)] += 1
    
    proj["exp_num_to_play"] = np.zeros(proj.shape[0], dtype=int)
    for i in exp_to_play:
        proj.at[i, "exp_num_to_play"] = exp_to_play[i]

    return proj["exp_num_to_play"].values


def predict_matchup(db: dbInterface, date: str, team1: str, team2: str, 
                    proj: pd.DataFrame = None, scores: pd.DataFrame = None,
                    kern_sig: float = GKERN_SIG, actual_played: bool = False):
    """
    Predicts the probability of victory for a matchup between the two teamIDs
    from the perspective of the morning of the specified date
    (i.e., no games that day have been played)

    Assumes that the exp_num_played column has already been created.

    Args:
        db (dbInterface): dbInterface object
        date (str): Date in YYYY-MM-DD format
        team1 (str): teamID of team1
        team2 (str): teamID of team2
        proj (pd.DataFrame, optional): Optionally, provide precomputed projections for speed
        scores (pd.DataFrame, optional): Optionally, provide precomputed scoreboard for speed
        actual_played (bool, optional): If reviewing past predictions, can use the 
                                        actual number of games played by each player,
                                        instead of the predicted number played.
        kern_sig (float, optional): STD of Gaussian kernal. Defaults to GKERN_SIG.
        actual_played (bool, optional): If reviewing past predictions, adjust the games
                                       played estimate to reflect the actual number of games
                                       played for each player. Can only select if proj is not
                                       provided, otherwise must do in proj.

    Returns:
        (overall p1 victory prob, overall p2 victory prob, overall tie),
        {stat: (p1 victory, p2 victory tie)},
        {outcome: probability}
        {stat: ((p1 amount projected, p1 amount currently),
               ((p2 amount projected, p2 amount currently))}
    """
    # Get proj if not given
    if proj is None:
        proj = proj_all_players(db, date, kern_sig, [team1, team2])
        final_date = db.week_date_range(db.week_for_date(date))[1]
        if actual_played:
            proj = adjust_for_actual_played(db, proj, date, final_date, kern_sig)

    # Get score
    if scores is None:
        week = db.week_for_date(date)
        scores = db.matchup_score(week, date)

    # Make projection dictionary to feed to prob_victory
    proj_dict = {}
    current_scores = {}
    proj_final = {}
    for col in utils.STATS_COLS:
        proj_dict[col] = (proj.loc[team1].apply(lambda row: row[col]*row["exp_num_to_play"], axis=1).sum(),
                          proj.loc[team2].apply(lambda row: row[col]*row["exp_num_to_play"], axis=1).sum())
        current_scores[col] = (scores.loc[team1][col],
                               scores.loc[team2][col])
        proj_final[col] = ((proj_dict[col][0], current_scores[col][0]),
                           (proj_dict[col][1], current_scores[col][1]))

    return prob_victory(proj_dict, current_scores) + (proj_final,)


def prob_victory(proj: dict[str, tuple[float]],
                 current_scores: dict[str, tuple[float]] = {"PTS": (0, 0), "FG3M": (0, 0), 
                                                            "REB": (0, 0), "AST": (0, 0),
                                                            "ST": (0, 0), "BLK": (0, 0),
                                                            "TO": (0, 0), "FGA": (0, 0),
                                                            "FGM": (0, 0), "FTA": (0, 0),
                                                            "FTM": (0, 0)}) -> (np.array,
                                                                                dict[str:tuple[float]]):
    """
    Returns the probability of victory in each category and overall
    between the two players

    Args:
        proj (dict): maps stat category to a tuple of player 1 and player 2 projected values
        current_scores (dict): maps stat category to a tuple of player 1 and player 2 values

    Returns: np.array and dict
        (overall p1 victory prob, overall p2 victory prob, overall tie),
        {stat: (p1 victory, p2 victory tie)},
        std for percent stats,
        {final score: probability}
    """

    simple_stats = ["PTS", "3PTM", "REB", "AST", "ST", "BLK", "TO"]
    percent_stats = {"FG%": ("FGA", "FGM"), "FT%": ("FTA", "FTM")}

    stat_victory = {}

    # Go through simple counting stats
    for stat in simple_stats:
        current_score = current_scores[stat]
        stat_victory[stat] = skellam_prob(proj[stat][0], proj[stat][1],
                                          current_score=current_score)
        # Reverse for TO
        if stat == "TO":
            w1 = stat_victory[stat][1]
            w2 = stat_victory[stat][0]
            stat_victory[stat] = (w1, w2, stat_victory[stat][2])

    # Go through percentage stats
    for stat in percent_stats:
        attempts = proj[percent_stats[stat][0]]
        made = proj[percent_stats[stat][1]]
        current_score = (current_scores[percent_stats[stat][0]],
                         current_scores[percent_stats[stat][1]])
        
        # TODO: Propagate out the std of the stats
        stat_victory[stat] = ratio_prob(attempts, made,
                                        current_score=current_score)


    # Loop through all 19,683 possible stat winning combinations/ties
    # iterate over all lists of 9 zeros (p1 victory) and ones (p2 victory), and twos (ties)
    probs = np.zeros(3)
    # Record probability of each outcome type (i.e, 5-4 victory, 4-4-1 tie)
    outcomes = {}
    # Pick win/loss/tie for each stat
    for combo in product(np.arange(3), repeat=9):
        # Classify the outcome
        players, wins = np.unique(combo, return_counts=True)
        players = list(players)
        key = []
        if 0 in players:
            key.append(wins[players.index(0)])
        else:
            key.append(0)
        if 1 in players:
            key.append(wins[players.index(1)]) 
        else:
            key.append(0) 
        if 2 in players:
            key.append(wins[players.index(2)])
        else:
            key.append(0)
        key = tuple(key)

        # Calculate probability of this outcome
        p = 1
        for i, stat in enumerate(stat_victory):
            p*=stat_victory[stat][combo[i]]
        if key in outcomes:
            outcomes[key] += p
        else:
            outcomes[key] = p

        # One player gets no wins
        if 1 not in players:
            probs[0]+=p
        elif 0 not in players:
            probs[1]+=p
        # Tie
        elif wins[0] == wins[1]:
            probs[2]+=p
        # Normal matchups
        else:
            if wins[0] > wins[1]:
                probs[0]+=p
            else:
                probs[1]+=p 
    
    # Normalize to smooth out numerical relics
    probs/=np.sum(probs)
    
    return probs, stat_victory, outcomes


def matchup_matrix(db: dbInterface, date: str, kern_sig: float = GKERN_SIG,
                   actual_played: bool = False):
    
    # Record stat projections for everyone
    total_proj = {}
    order = []

    # Precompute projected stats
    proj = proj_all_players(db, date, kern_sig, actual_played=actual_played)

    # Get Scoreboard
    scoreboard = db.matchup_score(db.week_for_date(date), date)
    managers = np.unique(scoreboard["teamID"])
    order = [db.teamID_lookup(x)[0] for x in managers]
    probMat = np.zeros((len(managers), len(managers)))
    for i1, m1 in enumerate(managers):
        for i2, m2 in enumerate(managers):
            if i2 > i1:
                prob, stats, outcomes, totals = predict_matchup(db, date, m1, m2,
                                                scores=scoreboard,
                                                proj=proj,
                                                kern_sig=kern_sig)

                # Record victory probability and total predicted stats
                probMat[i1, i2] = prob[0]
                probMat[i2, i1] = prob[1]
                if m1 not in total_proj:
                    entry = {}
                    for stat in totals:
                        entry[stat] = totals[stat][0]
                    total_proj[m1] = entry
                if m2 not in total_proj:
                    entry = {}
                    for stat in totals:
                        entry[stat] = totals[stat][1]
                    total_proj[m2] = entry
            elif i2 == i1:
                probMat[i1, i2] = np.nan
    
    to_df = []
    for teamID in total_proj:
        entry = total_proj[teamID]
        entry["teamID"] = teamID
        to_df.append(entry)
    total_proj = pd.DataFrame(to_df)
    total_proj.index = total_proj.teamID
    return probMat, order, total_proj


### NOTE: NOT CURRENTLY WORKING RIGHT
def assign_roster_spots(players: pd.DataFrame,
                        roster_spots: dict[str:int] = utils.ROSTER_SPOTS) -> dict[str:list[str]]:
    """
    Assigns roster spots, giving preference to players with more
    expected fantasy points in the case of a tie

    Args:
        players (pd.DataFrame): Dataframe with columns name, eligible_positions,
                                and appropriate stats. Returned from proj_all_players 

        roster_spots (dict, optional): Dictionary that says how many roster spots
                                       are avaliable for each position

    Returns:
        dict: maps position to a list of player names
    """

    def place_player(roster, eligible_positions):
        for pos in eligible_positions:
            if len(roster[pos]) < roster_spots[pos]:
                return pos
        return "BN"


    # Calculate fantasy points for deciding who plays
    # it's not there yet
    if "NBA_FANTASY_PTS" not in players.columns:
        players['NBA_FANTASY_PTS'] = utils.calc_fantasy_points(players)
    players = players.sort_values(by="NBA_FANTASY_PTS", ascending=False)
    
    # Assign roster spots by placing a player in their preferred
    # position first (in order by eligible_positions).
    #
    # If that position is already full, then the spot goes to the
    # player with the higher expected fantasy points and the 
    # other player is reassigned down the list.
    #
    # In the case of a player being placed on the bench, you go back around
    # and see if any other players that were given preference could be moved.
    #
    # This ends with the lowest fantasy point players being benched
    # in case of too many people
    #
    roster = {}
    for pos in roster_spots:
        roster[pos] = []
    roster["BN"] = []
    roster["ACTIVE"] = []
    elig_pos = {}
    # Interpret list and remove IL
    for i, row in players.iterrows():
        elig_pos[row["name"]] = [x for x in literal_eval(row["eligible_positions"]) if "IL" not in x]

    # Try to place each player
    for i in range(players.shape[0]):
        placed = False
        row = players.iloc[i]
        placement = place_player(roster, elig_pos[row["name"]])
        if placement != "BN":
            placed = True
            roster[placement].append(row["name"])
        else:
        # If would be place on bench,
        # Check to see if another player can be moved to keep
        # everyone active
            for pos in elig_pos[row["name"]]:
                for name2 in roster[pos]:
                    replacement = place_player(roster, elig_pos[name2])
                    if replacement != "BN":
                        roster[pos].remove(name2)
                        roster[pos].append(row["name"])
                        roster[replacement].append(name2)
                        placed = True
                        break
                if placed:
                    break


        if placed:
            roster["ACTIVE"].append(row["name"])
        if not placed:
            roster["BN"].append(row["name"])
    
    return roster


if __name__ == "__main__":

    

    
    db = dbInterface("past_season_dbs/yahoo_fantasy_2022_23.sqlite")
    date = "2023-01-30"
    # date2 = "2023-02-05"
    date = "2023-11-06"
    # date2 = "2023-02-05"

    week = db.week_for_date(date)
    scores = db.matchup_score(week, date)
    import time
    t0 = time.time()
    # db.player_stats("Paul George")
    proj = proj_all_players(db, date)
    # calc_multiplier(proj)
    # proj = adjust_for_actual_played(db, proj, date, db.week_date_range(week)[1])
    t1 = time.time()
    # db.get_nba_stats("WHERE PLAYER_NAME LIKE 'Paul George'")
    # entries = calc_actual_played(db, "418.l.20454.t.4", date, date2)

    res = predict_matchup(db, date, "428.l.2784.t.6", "428.l.2784.t.8", proj, scores)

    t2 = time.time()

    
    print(t1-t0)

    print(t2-t1)

    


    # sc, gm, curLg = refresh_oauth_file(oauthFile = 'yahoo_oauth.json')

    # week = curLg.current_week()

    # # proj, matchup_df = past_preds( sc, gm, curLg, week-1, savename="past_preds.png")

    # # run_predictions(sc, gm, curLg, week, "predictions")
    
    # ## TODO: FIX PLAYED TODAY OR NOT
    # players = get_all_taken_players_extra(sc, curLg, week, include_today=False)


    # matchup_df = extract_matchup_scores(curLg, week, nba_cols=True)
    # # matchup_df = None
    # stats = return_all_taken_stats(curLg, tp=players)
    
    # proj = project_stats_team(players, stats, subtract_played=True)

    # # # p, s, m  = prob_victory(proj, "Eli", "Chi Yen")    
    # plot_matchup_summary(proj, "Eli", "Jack", matchup_df=matchup_df)
    # plot_matchup_summary(proj, "David", "Gary", matchup_df=matchup_df)
    # # plot_matchup_summary(proj, "Fabio", "Yi Sheng", matchup_df=matchup_df)
    # # plot_matchup_summary(proj, "Fabio", "Yi Sheng", matchup_df=matchup_df)

    # # # # proj = project_stats_team(players, stats, num_games=4,count_IL=False, consider_status=False)
    # # # # probMat = ideal_matrix(proj, num_games=None, savename="actual_last_week.png")
    # probMat = ideal_matrix(proj, num_games=None, savename=f"preds_{TODAY}.png", matchup_df=matchup_df, week=week)



    