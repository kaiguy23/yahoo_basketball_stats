import datetime

import sys
sys.path.append("../")

import db_interface, utils


TEST_FILE = "../yahoo_save.sqlite"
TEST_SEASON = "2022_23"


TEST_FILE = "../yahoo_save.sqlite"
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
    assert utils.find_closest_date("2022-12-02", dates) == 2
    assert utils.find_closest_date("2022-12-03", dates) == 3
    assert utils.find_closest_date("2022-12-04", dates) == 3
    assert utils.find_closest_date("2022-12-05", dates) == 3
    assert utils.find_closest_date("2023-12-05", dates) == 4
