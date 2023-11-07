import datetime

import sys
sys.path.append("../")

import db_interface, utils

# TODO: REDO TESTS BASED ON NEW UPTO DATES FOR GAMES IN WEEK AND MATCHUP SCORE

TEST_FILE = "../past_season_dbs/yahoo_fantasy_2022_23.sqlite"
TEST_SEASON = "2022_23"
db = db_interface.dbInterface(TEST_FILE, TEST_SEASON)


def test_week_for_date():

    date_str = "2023-03-27"
    week = -1
    assert db.week_for_date(date_str) == week
    dt_obj = datetime.datetime.strptime(date_str, utils.DATE_SCHEMA)
    assert db.week_for_date(dt_obj) == week

    date_str = "2022-07-14"
    week = -1
    assert db.week_for_date(date_str) == week
    dt_obj = datetime.datetime.strptime(date_str, utils.DATE_SCHEMA)
    assert db.week_for_date(dt_obj) == week

    date_str = "2023-03-26"
    week = 22
    assert db.week_for_date(date_str) == week
    dt_obj = datetime.datetime.strptime(date_str, utils.DATE_SCHEMA)
    assert db.week_for_date(dt_obj) == week

    date_str = "2023-03-20"
    week = 22
    assert db.week_for_date(date_str) == week
    dt_obj = datetime.datetime.strptime(date_str, utils.DATE_SCHEMA)
    assert db.week_for_date(dt_obj) == week

    date_str = "2023-03-06"
    week = 20
    assert db.week_for_date(date_str) == week
    dt_obj = datetime.datetime.strptime(date_str, utils.DATE_SCHEMA)
    assert db.week_for_date(dt_obj) == week

    date_str = "2023-01-23"
    week = 15
    assert db.week_for_date(date_str) == week
    dt_obj = datetime.datetime.strptime(date_str, utils.DATE_SCHEMA)
    assert db.week_for_date(dt_obj) == week

    date_str = "2022-11-29"
    week = 7
    assert db.week_for_date(date_str) == week
    dt_obj = datetime.datetime.strptime(date_str, utils.DATE_SCHEMA)
    assert db.week_for_date(dt_obj) == week


def test_week_date_range():

    assert db.week_date_range(22) == ("2023-03-20", "2023-03-26")
    
    assert db.week_date_range(23) == -1
    
    assert db.week_date_range(7) == ("2022-11-28", "2022-12-04")


def test_player_affiliation():

    assert db.player_affiliation("Markelle Fultz", "2022-12-01") == ("418.l.20454.t.8",
                                                                     "ORL")
    
    assert db.player_affiliation("Chet Holmgren", "2023-02-01") == ("418.l.20454.t.9",
                                                                     "OKC")
    
    assert db.player_affiliation("O.G. Anunoby", "2023-03-05") == ("418.l.20454.t.4",
                                                                     "TOR")
    assert db.player_affiliation("Miles Bridges", "2023-03-05") == ("", "N/A")


def test_player_stats():

    assert len(db.player_stats("O.G. Anunoby")) > 0
    assert len(db.player_stats("Chet Holmgren")) == 0

    stats = db.player_stats("Domantas Sabonis")
    stats = stats[stats["GAME_DATE"] == "2023-02-06"].iloc[0]

    assert abs(stats["FGM"]/stats["FGA"] - 0.700) < 0.001
    assert abs(stats["FTM"]/stats["FTA"] - 1.000) < 0.001
    assert stats["3PTM"] == 1
    assert stats["PTS"] == 17
    assert stats["REB"] == 7
    assert stats["AST"] == 10
    assert stats["ST"] == 2
    assert stats["BLK"] == 0
    assert stats["TO"] == 2


def test_teamID_lookup():

    assert db.teamID_lookup("418.l.20454.t.8") == ("Eli", "Hartless 💔")


def test_manager_to_teamID():

    assert db.manager_to_teamID("Eli") == "418.l.20454.t.8"


def test_games_in_week():

    assert db.games_in_week("DET", 14) == 1
    assert db.games_in_week("BKN", 1) == 2
    assert db.games_in_week("NOP", 20) == 4
    
    assert db.games_in_week("DET", 7, "2022-11-28")[0:2] == (0, 3)

    assert db.games_in_week("ATL", 7, "2022-11-28")[0:2] == (0, 3)
    assert db.games_in_week("ATL", 7, "2022-11-29")[0:2] == (1, 2)
    assert db.games_in_week("ATL", 7, "2022-11-30")[0:2] == (1, 2)
    assert db.games_in_week("ATL", 7, "2022-12-01")[0:2] == (2, 1)
    assert db.games_in_week("ATL", 7, "2022-12-02")[0:2] == (2, 1)
    assert db.games_in_week("ATL", 7, "2022-12-03")[0:2] == (3, 0)
    assert db.games_in_week("ATL", 7, "2022-12-04")[0:2] == (3, 0)
    
    remaining = db.games_in_week("ATL", 7, "2022-11-28")[2]
    for pair in zip(remaining, [1, 0, 1, 0, 1, 0, 0]):
        assert pair[0] == pair[1]

def test_matchup_score():

    scores = db.matchup_score(7)
    scores = scores[scores.manager.isin(["Eli"])].iloc[0]
    assert scores["FGM"] == 281
    assert scores["FGA"] == 535
    assert scores["FTM"] == 119
    assert scores["FTA"] == 161
    assert abs(scores["FG%"] - 0.525) < 0.001
    assert abs(scores["FT%"] - 0.739) < 0.001
    assert scores["3PTM"] == 67
    assert scores["PTS"] == 748
    assert scores["REB"] == 288
    assert scores["AST"] == 165
    assert scores["ST"] == 44
    assert scores["BLK"] == 34
    assert scores["TO"] == 97

    # After one day
    scores = db.matchup_score(17, "2023-02-07")
    scores = scores[scores.manager.isin(["Gary"])].iloc[0]
    assert abs(scores["FG%"] - 0.522) < 0.001
    assert abs(scores["FT%"] - 0.913) < 0.001
    assert scores["3PTM"] == 12
    assert scores["PTS"] == 129
    assert scores["REB"] == 41
    assert scores["AST"] == 25
    assert scores["ST"] == 8
    assert scores["BLK"] == 2
    assert scores["TO"] == 15

    # After zero days
    scores = db.matchup_score(16, "2023-01-30")
    scores = scores[scores.manager.isin(["Gary"])].iloc[0]
    assert scores["FG%"] == 0
    assert scores["FT%"] == 0
    assert scores["3PTM"] == 0
    assert scores["PTS"] == 0
    assert scores["REB"] == 0
    assert scores["AST"] == 0
    assert scores["ST"] == 0
    assert scores["BLK"] == 0
    assert scores["TO"] == 0

    # Last day of week
    scores = db.matchup_score(17, "2023-02-12")
    scores = scores[scores.manager.isin(["Fabio"])].iloc[0]
    assert scores["3PTM"] == 59-9
    assert scores["PTS"] == 595-40
    assert scores["REB"] == 205-4
    assert scores["AST"] == 125-8
    assert scores["ST"] == 37-2
    assert scores["BLK"] == 24-1
    assert scores["TO"] == 64-4
