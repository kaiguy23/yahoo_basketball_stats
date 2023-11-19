import pytest

import sys
sys.path.append("../")

import db_builder
import db_interface


TEST_FILE = "../past_season_dbs/yahoo_fantasy_2022_23.sqlite"
TEST_SEASON = "2022_23"
db = db_interface.dbInterface(TEST_FILE, TEST_SEASON)

def test_update_fantasy_schedule():
    assert True

def test_update_nba_stats():
    assert True

def test_update_nba_rosters():
    assert True

def test_update_num_games_per_day():
    assert True

def test_update_nba_schedule():
    assert True

def test_update_fantasy_teams():
    assert True

def test_update_fantasy_rosters():
    assert True