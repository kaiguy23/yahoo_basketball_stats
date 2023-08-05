import pandas as pd
import numpy as np
import sqlite3
import datetime

from typing import Union

import utils


class dbInterface:

    def __init__(self, f: str, season: str = utils.DEFAULT_SEASON):
        """
        Makes an object to read the yahoo fantasy database

        Args:
            f (str): Database file
            season (str, optional): String for the season, like 2022_23.
                                    Defaults to utils.DEFAULT_SEASON.
        """

        # Open in read only mode
        self.con = sqlite3.connect(f, uri=True)
        self.cur = self.con.cursor()
        self.weeks = None
        self.fantasy_lookup = None
        self.nba_stats = None
        self.nba_rosters = None
        self.fantasy_teams = None
        self.games_per_day = None
        self.season = season

    def __del__(self):
        self.con.close()

    @staticmethod
    def build_select_statement(table_name: str,
                               filter_statement: str = "") -> str:
        """
        Formats a select statement to query the database

        Args:
            table_name (str): table to read from
            filter_statement (str, optional): SQL filter statement to be added.
                                              e.g. "WHERE manager LIKE 'Eli'"

        Returns:
            str: desired select statement
        """
        query = f"SELECT * FROM {table_name}"
        if filter_statement != "":
            query += " " + filter_statement
        return query

    def table_names(self) -> list:
        """
        Returns a list of the tables present in the database file

        Returns:
            list: list of tables
        """

        return self.cur.execute("SELECT name FROM sqlite_master\
                                WHERE type='table'").fetchall()

    def get_fantasy_schedule(self, filter_statement: str = "") -> pd.DataFrame:
        """
        Reads the fantasy schedule table

        Args:
            filter_statement (str, optional): SQL filter statement to be added.
                                              e.g. "WHERE manager LIKE 'Eli'"

        Returns:
            pd.DataFrame: fantasy schedule table
        """
        table_name = f"FANTASY_SCHEDULE_{self.season}"
        query = dbInterface.build_select_statement(table_name,
                                                   filter_statement)
        return pd.read_sql_query(query, self.con)

    def get_nba_schedule(self, filter_statement: str = "") -> pd.DataFrame:
        """
        Reads the nba schedule table

        Args:
            filter_statement (str, optional): SQL filter statement to be added.
                                              e.g. "WHERE manager LIKE 'Eli'"

        Returns:
            pd.DataFrame: nba schedule table
        """
        table_name = f"NBA_SCHEDULE_{self.season}"
        query = dbInterface.build_select_statement(table_name,
                                                   filter_statement)
        return pd.read_sql_query(query, self.con)

    def get_fantasy_rosters(self, filter_statement: str = "") -> pd.DataFrame:
        """
        Reads the fantasy rosters table

        Args:
            filter_statement (str, optional): SQL filter statement to be added.
                                              e.g. "WHERE manager LIKE 'Eli'"

        Returns:
            pd.DataFrame: fantasy rosters table
        """
        table_name = f"FANTASY_ROSTERS_{self.season}"
        query = dbInterface.build_select_statement(table_name,
                                                   filter_statement)
        return pd.read_sql_query(query, self.con)

    def get_nba_stats(self, filter_statement: str = "") -> pd.DataFrame:
        """
        Reads the nba stats table
        Args:
            filter_statement (str, optional): SQL filter statement to be added.
                                              e.g. "WHERE manager LIKE 'Eli'"

        Returns:
            pd.DataFrame: player stats table
        """
        table_name = f"NBA_STATS_{self.season}"
        query = dbInterface.build_select_statement(table_name,
                                                   filter_statement)
        return pd.read_sql_query(query, self.con)

    def get_games_per_day(self, filter_statement: str = "") -> pd.DataFrame:
        """
        Reads the games per week table

        Args:
            filter_statement (str, optional): SQL filter statement to be added.
                                              e.g. "WHERE manager LIKE 'Eli'"

        Returns:
            pd.DataFrame: games per week table
        """
        table_name = f"GAMES_PER_DAY_{self.season}"
        query = dbInterface.build_select_statement(table_name,
                                                   filter_statement)
        df = pd.read_sql_query(query, self.con)
        df.index = df['date']
        return df

    def get_nba_rosters(self, filter_statement: str = "") -> pd.DataFrame:
        """
        Reads the nba rosters table

        Args:
            filter_statement (str, optional): SQL filter statement to be added.
                                              e.g. "WHERE manager LIKE 'Eli'"

        Returns:
            pd.DataFrame: nba rosters table
        """
        table_name = f"NBA_ROSTERS_{self.season}"
        query = dbInterface.build_select_statement(table_name,
                                                   filter_statement)
        df = pd.read_sql_query(query, self.con)
        return df

    def get_fantasy_teams(self) -> pd.DataFrame:
        """
        Reads the current fantasy teams table

        Returns:
            pd.DataFrame: current fantasy teams table
        """
        table_name = "CURRENT_FANTASY_TEAMS"
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)

    def week_for_date(self, date: Union[str, datetime.datetime]) -> int:
        """
        Tells you which yahoo week number the specified date is in
        returns -1 if it's not in any week

        Args:
            date (Union[str, datetime.datetime]): date string in the
                                                  standard scheme or 
                                                  datetime object.

        Returns:
            int: Yahoo week number
        """
        # Build the weeks dataframe if it hasn't been built yet
        if self.weeks is None:
            fantasy_schedule = self.get_fantasy_schedule()
            self.weeks = fantasy_schedule[fantasy_schedule.teamID ==
                                          fantasy_schedule.teamID.iloc[-1]]
            self.weeks = self.weeks[['week', 'startDate', 'endDate']]
            self.weeks.index = self.weeks.week
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, utils.DATE_SCHEMA)

        for week, row in self.weeks.iterrows():
            start_day = datetime.datetime.strptime(row['startDate'],
                                               utils.DATE_SCHEMA)
            end_day = datetime.datetime.strptime(row['endDate'],
                                             utils.DATE_SCHEMA)
            if start_day <= date <= end_day:
                return week
        
        return -1
            
    def week_date_range(self, week: int,
                        season: str = utils.DEFAULT_SEASON) -> tuple:
        """
        Returns the start and end date of the specified week

        Args:
            week (int): Yahoo week in the season

        Returns:
           tuple: (first day, last day)
        """

        # Build the weeks dataframe if it hasn't been built yet
        if self.weeks is None:
            fantasy_schedule = self.get_fantasy_schedule()
            self.weeks = fantasy_schedule[fantasy_schedule.teamID ==
                                          fantasy_schedule.teamID.iloc[-1]]
            self.weeks = self.weeks[['week', 'startDate', 'endDate']]
            self.weeks.index = self.weeks.week

        # Return the start and end dates
        return (self.weeks.at[week, 'startDate'],
                self.weeks.at[week, 'endDate'])

    def player_affiliation(self, name: str,
                           date: Union[str, datetime.datetime]) -> tuple:
        """
        Returns the affiliations for the given NBA player
        on the given date

        Args:
            name (str): name of nba player (from NBA API)
            date (str or datetime): date to look up for

        Returns:
           Tuple: (yahoo fantasy team id, nba team name)
        """

        if isinstance(date, str):
            date = datetime.datetime.strptime(date, utils.DATE_SCHEMA)
        date_str = date.strftime(utils.DATE_SCHEMA)

        # Generate lookups if they haven't been made yet
        if self.fantasy_lookup is None:
            self.fantasy_lookup = self.get_fantasy_rosters().groupby("name")
        if self.nba_stats is None:
            self.nba_stats = self.get_player_stats().groupby("PLAYER_NAME")
        if self.nba_rosters is None:
            self.nba_rosters = self.get_nba_rosters().groupby("PLAYER_NAME")

        # Find which NBA team the player was on for the specified date
        if name in self.nba_stats.indices:
            entries = self.nba_stats.get_group(name)
            closest_i = find_closest_date(date, entries["GAME_DATE"].values)
            nba_team = entries.iloc[closest_i]["TEAM_ABBREVIATION"]
        else:
            entries = self.nba_rosters.get_group(name)
            closest_i = find_closest_date(date, entries["DATE"].values)
            nba_team = entries.iloc[closest_i]["TEAM_ABBREVIATION"]

        # Check if the player was on a fantasy roster that day
        fantasy_entries = self.fantasy_lookup.get_group(name)
        from_date = fantasy_entries[fantasy_entries['date'] == date_str]
        if from_date.shape[0] > 0:
            fantasy_team = from_date.iloc[0]['teamID']
        else:
            fantasy_team = ""

        return (fantasy_team, nba_team)

    def player_stats(self, name: str) -> pd.DataFrame:
        """
        Returns all the entries from the nba stats table
        that correspond to the specified player.

        Args:
            name (str): name of player from nba api

        Returns:
            pd.DataFrame: dataframe of games played
        """

        # Generate lookups if they haven't been made yet
        if self.nba_stats is None:
            self.nba_stats = self.get_player_stats().groupby("PLAYER_NAME")
        
        return self.nba_stats[name]


    def teamID_lookup(self, teamID: str) -> tuple:
        """
        Gets the manager and team names, given
        the Yahoo team id

        Args:
            teamID (str): Yahoo unique identifier for the team

        Returns:
            tuple: (manager name, team name)
        """

        # Build the lookup table
        if self.fantasy_teams is None:
            self.fantasy_teams = self.get_fantasy_teams()
            self.fantasy_teams.index = self.fantasy_teams.teamID

        return (self.fantasy_teams.at[teamID, 'manager'],
                self.fantasy_teams.at[teamID, 'teamName'])

    def games_in_week(self, nba_team: str, week: int,
                      upto: Union[str, datetime.datetime] = "") -> Union[int, tuple]:
        """
        Returns the number of games in the given fantasy
        week for the given nba team

        Args:
            nba_team (str): NBA team 3 letter abbreviation
                            i.e., GSW
            week (int): Yahoo week in the season
            upto (str or datetime): instead return games
                                    before and after the given
                                    date (upto date is in before)

        Returns:
            int/tuple: number of games or (n_games before upto,
                                           n_games after upto)
        """
        # Check upto is in the right week
        if isinstance(upto, str):
            if upto != "":
                if not self.week_for_date(upto) == week:
                    raise(ValueError, "upto not in correct week")
        elif not self.week_for_date(upto) == week:
            raise(ValueError, "upto not in correct week")
        
        # Load table
        if self.games_per_day is None:
            self.games_per_day = self.get_games_per_day()

        start_day_str, end_day_str = self.week_date_range(week)
        start_day = datetime.datetime.strptime(start_day_str,
                                               utils.DATE_SCHEMA)
        end_day = datetime.datetime.strptime(end_day_str,
                                             utils.DATE_SCHEMA)

        if isinstance(upto, str) and upto != "":
            upto_dt = datetime.datetime.strptime(upto, utils.DATE_SCHEMA)
        elif upto == "":
            upto_dt = end_day
        elif isinstance(upto, datetime.datetime):
            upto_dt = upto

        counts = [0, 0]
        for date in pd.date_range(start_day, end_day, freq='D'):
            date_str = date.strftime(utils.DATE_SCHEMA)
            if date <= upto_dt:
                counts[0] += self.games_per_day.at[date_str, nba_team]
            else:
                counts[1] += self.games_per_day.at[date_str, nba_team]

        if isinstance(upto, str):
            if upto == "":
                return counts[0]
            else:
                return counts
        else:
            return counts

    def matchup_score(self, week: int,
                      upto: Union[str, datetime.datetime] = ""):
        """
        Returns the matchup score for a given week,
        optionally upto (including) a date within that
        week

        Args:
            nba_team (str): NBA team 3 letter abbreviation
                            i.e., GSW
            week (int): Yahoo week in the season
            upto (str or datetime): instead return games
                                    before and after the given
                                    date (upto date is in before)

        Returns:
            int: number of games
        """
        # Check upto is in the right week
        if isinstance(upto, str):
            # Convert from string to date
            if upto != "":
                if not self.week_for_date(upto) == week:
                    raise(ValueError, "upto not in correct week")
                else:
                    upto = datetime.datetime.strptime(upto,
                                                      utils.DATE_SCHEMA)
            else:
                upto = datetime.datetime.strptime(self.week_date_range(week)[1],
                                                  utils.DATE_SCHEMA)

        elif not self.week_for_date(upto) == week:
            raise(ValueError, "upto not in correct week")

        # Get all stats entries from the week where
        # people actually played
        stats = self.get_nba_stats(f"WHERE week = {week}")

        # Filter based on date
        to_keep = [datetime.datetime.strptime(x, utils.DATE_SCHEMA) <= upto
                   for x in stats['GAME_DATE']]
        stats = stats[to_keep]
        stats = stats[stats["selected_position"].isin(utils.ACTIVE_POS)]
        stats = stats.groupby("teamID")

        # Get fantasy schedule for the week
        sched = self.get_fantasy_schedule(f"WHERE week = {week}")
        sched.index = sched.teamID


        for team in sched.teamID:
            team_stats = stats.get_group(team)
            for stat in utils.STATS_COLS:
                sched.at[team, stat] = team_stats[stat].sum()
            for stat in utils.PERC_STATS:
                attempts = stat.replace("%", "A")
                made = stat.replace("%", "M")
                sched.at[team, stat] = (sched.at[team, made] /
                                        sched.at[team, attempts])

        return sched


        # Maybe modify table to be games per day
    

## TODO: Make games played this week so far function using unique values of (date, team) in player_stats

## TODO: Make a function that gets the matchup score on any given day by summing player stats




def find_closest_date(d: Union[str, datetime.datetime],
                      dates: list):
    """
    Finds the index of the closest date in
    dates to the date d. If d/dates
    are strings, assumes they are in the default
    date format.

    Args:
        d (str or datetime): date to find the closest entry to
        dates (list of str or datetime): list of dates to compare to
    """

    if isinstance(d, str):
        d = datetime.datetime.strptime(d, utils.DATE_SCHEMA)

    closest_i = 0
    diff = np.inf
    for i in range(len(dates)):
        d2 = dates[i]
        if isinstance(d2, str):
            d2 = datetime.datetime.strptime(d2, utils.DATE_SCHEMA)
        diff2 = abs((d - d2).days)
        if diff2 < diff:
            diff = diff2
            closest_i = i

    return closest_i


if __name__ == "__main__":
    db = dbInterface("yahoo_save.sqlite")
