import pytest

import sys
sys.path.append("../")

import db_interface


TEST_FILE = "../yahoo_save.sqlite"
TEST_SEASON = "2022_23"


TEST_FILE = "../yahoo_save.sqlite"
TEST_SEASON = "2022_23"
db = db_interface.dbInterface(TEST_FILE, TEST_SEASON)

def test_week_for_date():
    assert False

def test_week_date_range():
    assert False

def test_player_affiliation():
    assert False

def test_player_stats():
    assert False

def test_teamID_lookup():
    assert False

def test_games_in_week():
    assert False

def test_matchup_score():
    assert False

def test_find_closest_date():
    assert False