import sys
sys.path.append("../")

import db_interface, utils, pred


TEST_FILE = "../past_season_dbs/yahoo_fantasy_2022_23.sqlite"
TEST_SEASON = "2022_23"
db = db_interface.dbInterface(TEST_FILE, TEST_SEASON)


def test_gkern1sided():
    assert False


def test_skellam_prob():
    assert False


def test_ratio_prob():
    assert False


def test_prob_victory():
    assert False

def test_calc_multiplier():
    date = "2023-02-13"

    proj = pred.proj_all_players(db, date)
    pred.calc_multiplier(proj)


def test_calc_actual_played():

    
    date = "2023-01-30"
    date2 = "2023-02-05"


    counts = pred.calc_actual_played(db, "418.l.20454.t.4", date, date2)
    assert sum(counts.values()) == 40
    assert counts["Jalen Suggs"] == 3
    
    counts = pred.calc_actual_played(db, "418.l.20454.t.4", date, date2)
    assert sum(counts.values()) == 46
