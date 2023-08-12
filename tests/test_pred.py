import sys
sys.path.append("../")

import db_interface, utils


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