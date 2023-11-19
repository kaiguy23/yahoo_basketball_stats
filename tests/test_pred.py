import sys
sys.path.append("../")

import numpy as np
import scipy as sp

import db_interface, utils, pred


TEST_FILE = "../past_season_dbs/yahoo_fantasy_2022_23.sqlite"
TEST_SEASON = "2022_23"
db = db_interface.dbInterface(TEST_FILE, TEST_SEASON)


def test_gkern1sided():
    
    kern = pred.gkern_1sided(10, 2)
    assert sum(kern) == 1
    assert kern[0]/kern[4] == 1/(np.exp(-0.5*(4**2)/(2**2)))



def test_skellam_prob():
    
    # both mu > 0
    mu1 = 1
    mu2 = 2
    w1, w2, tie = pred.skellam_prob(mu1, mu2, (0, 0))
    test = [0, 0, 0]
    for i in range(-100, 100):
        val = sp.stats.skellam.pmf(i, mu1, mu2)
        if i == 0:
            test[2] += val
        elif i < 0:
            test[1] += val
        else:
            test[0] += val
    epsilon = 0.001
    assert np.abs(w1-test[0]) < epsilon
    assert np.abs(w2-test[1]) < epsilon
    assert np.abs(tie-test[2]) < epsilon

    mu1 = 2
    mu2 = 2
    w1, w2, tie = pred.skellam_prob(mu1, mu2, (2, 2))
    assert w1 == w2


    # mu2 = 0
    mu1 = 2
    mu2 = 0
    w1, w2, tie = pred.skellam_prob(mu1, mu2, (2, 1))
    assert w1 == 1
    assert w2 == 0
    assert tie == 0

    # mu1 = 0
    mu1 = 0
    mu2 = 2
    w1, w2, tie = pred.skellam_prob(mu1, mu2, (1, 2))
    assert w1 == 0
    assert w2 == 1
    assert tie == 0

def test_ratio_prob():
    
    # Evenly matched
    attempts = (50, 50)
    made = (20, 20)
    current_score = ((10, 10), (10, 10))
    w1, w2, tie = pred.ratio_prob(attempts, made,
                                  current_score=current_score)
    assert np.abs(w1-w2) < 0.05

    # Unevenly matched
    attempts = (50, 50)
    made = (30, 20)
    current_score = ((10, 10), (9, 5))
    w1, w2, tie = pred.ratio_prob(attempts, made, current_score=current_score)
    assert w1 > 0.95

    # Impossible
    attempts = (50, 0)
    made = (30, 0)
    current_score = ((10, 10), (9, 0))
    w1, w2, tie = pred.ratio_prob(attempts, made, current_score=current_score)
    assert w1 == 1

    attempts = (0, 30)
    made = (0, 20)
    current_score = ((10, 10), (0, 5))
    w1, w2, tie = pred.ratio_prob(attempts, made, current_score=current_score)
    assert w2 == 1

    attempts = (0, 0)
    made = (0, 0)
    current_score = ((10, 10), (5, 5))
    w1, w2, tie = pred.ratio_prob(attempts, made, current_score=current_score)
    assert tie == 1

    attempts = (0, 0)
    made = (0, 0)
    current_score = ((10, 10), (4, 5))
    w1, w2, tie = pred.ratio_prob(attempts, made, current_score=current_score)
    assert w2 == 1

    attempts = (0, 0)
    made = (0, 0)
    current_score = ((10, 10), (4, 3))
    w1, w2, tie = pred.ratio_prob(attempts, made, current_score=current_score)
    assert w1 == 1


# def test_prob_victory():
#     assert False

def test_calc_multiplier():

    # Fabio/Chi Yen week 19
    date = "2023-03-05"
    proj = pred.proj_all_players(db, date)
    pred.calc_multiplier(proj)
    assert proj.loc["418.l.20454.t.1"].exp_num_to_play.sum() == 10
    assert proj.loc["418.l.20454.t.9"].exp_num_to_play.sum() == 7


def test_calc_actual_played():

    
    date = "2023-01-30"
    date2 = "2023-02-05"

    # Eli Jack Week 16 Matchup
    counts = pred.calc_actual_played(db, "418.l.20454.t.4", date, date2)
    assert sum(counts.values()) == 39
    # Jalen Suggs had a DNP coaches decision making 40 -> 39
    assert counts["Jalen Suggs"] == 2
    
    counts = pred.calc_actual_played(db, "418.l.20454.t.8", date, date2)
    assert sum(counts.values()) == 46

    # Fabio and Chi Yen week 19
    date = "2023-03-05"
    date2 = "2023-03-05"
    counts = pred.calc_actual_played(db, "418.l.20454.t.1", date, date2)
    for player in ["Delon Wright", "Kris Dunn", "Tyus Jones", "Jalen Williams",
                   "Charles Bassey", "Devin Booker", "Jaylin Williams",
                   "Mark Williams", "Josh Giddey", "Xavier Tillman"]:
        assert counts[player] == 1
    for player in ["RJ Barret", "Jaden Ivey", "James Wiseman",
                   "Karl-Anthony Towns", "P.J. Washington"]:
        assert player not in counts

    date = "2023-02-27"
    date2 = "2023-03-05"
    counts = pred.calc_actual_played(db, "418.l.20454.t.1", date, date2)
    assert sum(counts.values()) == 46



    date = "2023-03-04"
    counts = pred.calc_actual_played(db, "418.l.20454.t.9", date, date2)
    assert sum(counts.values()) == 8
    for player in counts:
        if player == "Brook Lopez":
            assert counts[player] == 2
        else:
            assert counts[player] == 1


def test_adjust_for_actual_played():

    
    date = "2023-01-30"
    date2 = "2023-02-05"

    proj = pred.proj_all_players(db, date)
    proj = pred.adjust_for_actual_played(db, proj, date, date2)

    # Eli Jack Week 16 Matchup
    assert proj.loc["418.l.20454.t.4"]["exp_num_to_play"].sum() == 39
    assert proj.loc[("418.l.20454.t.4", "Jalen Suggs")]["exp_num_to_play"] == 2
    assert proj.loc["418.l.20454.t.8"]["exp_num_to_play"].sum() == 46


    # Fabio Chi Yen Week 19 Matchup
    date = "2023-03-05"
    date2 = "2023-03-05"
    proj = pred.proj_all_players(db, date)
    proj = pred.adjust_for_actual_played(db, proj, date, date2)
    assert proj.loc["418.l.20454.t.1"]["exp_num_to_play"].sum() == 10
    for player in ["Delon Wright", "Kris Dunn", "Tyus Jones", "Jalen Williams",
                   "Charles Bassey", "Devin Booker", "Jaylin Williams",
                   "Mark Williams", "Josh Giddey", "Xavier Tillman"]:
        assert proj.loc[("418.l.20454.t.1", player)]["exp_num_to_play"] == 1
    for player in ["RJ Barret", "Jaden Ivey", "James Wiseman",
                   "Karl-Anthony Towns", "P.J. Washington"]:
        if ("418.l.20454.t.1", player) in proj:
            assert proj.loc[("418.l.20454.t.1", player)] == 0

    date = "2023-03-04"
    proj = pred.proj_all_players(db, date)
    proj = pred.adjust_for_actual_played(db, proj, date, date2)
    assert proj.loc["418.l.20454.t.9"]["exp_num_to_play"].sum() == 8
    assert proj.loc[("418.l.20454.t.9", "Brook Lopez")]["exp_num_to_play"] == 2
    assert proj.loc[("418.l.20454.t.9", "Buddy Hield")]["exp_num_to_play"] == 1


if __name__ == "__main__":

    test_calc_actual_played()
    test_adjust_for_actual_played()