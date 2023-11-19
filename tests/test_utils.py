import datetime

import sys
sys.path.append("../")

import db_interface, utils


TEST_FILE = "../past_season_dbs/yahoo_fantasy_2022_23.sqlite"
TEST_SEASON = "2022_23"
db = db_interface.dbInterface(TEST_FILE, TEST_SEASON)


def test_find_closest_date():
    dates = ["2022-11-28", "2022-11-29",
             "2022-11-30", "2022-12-04",
             "2023-03-21"]
    
    assert utils.find_closest_date("2022-11-27", dates) == 0
    assert utils.find_closest_date("2022-11-28", dates) == 0
    assert utils.find_closest_date("2022-11-29", dates) == 1
    assert utils.find_closest_date("2022-11-30", dates) == 2
    assert utils.find_closest_date("2022-12-01", dates) == 2
    assert utils.find_closest_date("2022-12-02", dates) in [2, 3]
    assert utils.find_closest_date("2022-12-03", dates) == 3
    assert utils.find_closest_date("2022-12-04", dates) == 3
    assert utils.find_closest_date("2022-12-05", dates) == 3
    assert utils.find_closest_date("2023-12-05", dates) == 4


def test_find_closest_date_fast():
    dates = ["2022-11-28", "2022-11-29",
             "2022-11-30", "2022-12-04",
             "2023-03-21"]
    
    assert utils.find_closest_date_fast("2022-11-27", dates) == 0
    assert utils.find_closest_date_fast("2022-11-28", dates) == 0
    assert utils.find_closest_date_fast("2022-11-29", dates) == 1
    assert utils.find_closest_date_fast("2022-11-30", dates) == 2
    assert utils.find_closest_date_fast("2022-12-01", dates) == 2
    assert utils.find_closest_date_fast("2022-12-02", dates) in [2, 3]
    assert utils.find_closest_date_fast("2022-12-03", dates) == 3
    assert utils.find_closest_date_fast("2022-12-04", dates) == 3
    assert utils.find_closest_date_fast("2022-12-05", dates) == 3
    assert utils.find_closest_date_fast("2023-12-05", dates) == 4

def test_yahoo_to_nba_name():

    assert utils.yahoo_to_nba_name("Nikola Jokic") == "Nikola Jokic"
    assert utils.yahoo_to_nba_name("OG Anunoby") == "O.G. Anunoby"
    assert utils.yahoo_to_nba_name("Yi Sheng Ong",
                                   {"Yi Sheng Ong": "Asian Jordan"}) == "Asian Jordan"
    
def test_matchip_winner():

    # Selected week 19 matchups
    scores = db.matchup_score(19)
    assert utils.matchup_winner(scores.loc["418.l.20454.t.8"],
                                scores.loc["418.l.20454.t.10"])[0] == [5, 4, 0]
    assert utils.matchup_winner(scores.loc["418.l.20454.t.8"],
                                scores.loc["418.l.20454.t.10"])[1] == 0
    assert utils.matchup_winner(scores.loc["418.l.20454.t.10"],
                                scores.loc["418.l.20454.t.8"])[1] == 1
    assert utils.matchup_winner(scores.loc["418.l.20454.t.8"],
                                scores.loc["418.l.20454.t.12"])[0] == [4, 4, 1]
    assert utils.matchup_winner(scores.loc["418.l.20454.t.8"],
                                scores.loc["418.l.20454.t.12"])[1] == 2



if __name__ == "__main__":
    test_find_closest_date()