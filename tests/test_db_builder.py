import pytest

import sys
sys.path.append("../")

import db_builder
import db_interface


TEST_FILE = "../yahoo_save.sqlite"
TEST_SEASON = "2022_23"
db = db_interface.dbInterface(TEST_FILE, TEST_SEASON)

def test_update_fantasy_schedule():
    assert False

def test_update_nba_stats():
    assert False

def test_update_nba_rosters():
    assert False

def test_update_num_games_per_day():
    assert False

def test_update_nba_schedule():
    assert False

def test_update_fantasy_teams():
    assert False

def test_update_fantasy_rosters():
    assert False