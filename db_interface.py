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

        # Generate lookups
        self.generate_lookups()

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
        table_name = f"FANTASY_TEAMS_{self.season}"
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)
    

    def generate_lookups(self) -> None:

        # Generate lookups if they haven't been made yet
        if self.fantasy_lookup is None:
            self.fantasy_lookup = self.get_fantasy_rosters().groupby("name")
        if self.nba_stats is None:
            self.nba_stats = self.get_nba_stats().groupby("PLAYER_NAME")
        if self.nba_rosters is None:
            self.nba_rosters = self.get_nba_rosters().groupby("PLAYER_NAME")

        # Build the weeks dataframe if it hasn't been built yet
        if self.weeks is None:
            fantasy_schedule = self.get_fantasy_schedule()
            self.weeks = fantasy_schedule[fantasy_schedule.teamID ==
                                          fantasy_schedule.teamID.iloc[-1]]
            self.weeks = self.weeks[['week', 'startDate', 'endDate']]
            self.weeks.index = self.weeks.week

        # Build the lookup table
        if self.fantasy_teams is None:
            self.fantasy_teams = self.get_fantasy_teams()
            self.fantasy_teams.index = self.fantasy_teams.teamID

        # Load table
        if self.games_per_day is None:
            self.games_per_day = self.get_games_per_day()

    
    def fantasy_free_agents(self, date: Union[str, datetime.datetime]) -> tuple[str]:
        """
        Returns a dataframe of fantasy free agents for a given day,

        i.e., players that are not on any team.

        Args:
            date (Union[str, datetime.datetime]): date in question

        Returns:
            tuple[str]: tuple of player names 
        """
        if not isinstance(date, str):
            date = date.strftime(utils.DATE_SCHEMA)

        on_roster = set(self.fantasy_rosters(date)["name"])
        all_players = set(list(self.nba_rosters.groups) + list(self.nba_stats.groups))
        return tuple([x for x in all_players if x not in on_roster])
    
    def fantasy_rosters(self, date: Union[str, datetime.datetime]) -> tuple[str]:
        """
        Returns a dataframe of fantasy rosters for a given day,

        i.e., players that are on a fantasy team

        Args:
            date (Union[str, datetime.datetime]): date in question

        Returns:
            tuple[str]: tuple of player names 
        """
        if not isinstance(date, str):
            date = date.strftime(utils.DATE_SCHEMA)
        return self.get_fantasy_rosters(f"WHERE date LIKE '{date}'")

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
                        season: str = utils.DEFAULT_SEASON) -> tuple[str]:
        """
        Returns the start and end date of the specified week.

        Returns -1 if the week is not present.

        Args:
            week (int): Yahoo week in the season

        Returns:
           tuple[str]: (first day, last day)
        """
        
        # Check if week is valid
        if week not in self.weeks.week:
            return -1

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

        # Find which NBA team the player was on for the specified date
        if name in self.nba_stats.indices:
            entries = self.nba_stats.get_group(name)
            closest_i = utils.find_closest_date(date, entries["GAME_DATE"].values)
            nba_team = entries.iloc[closest_i]["TEAM_ABBREVIATION"]
        else:
            entries = self.nba_rosters.get_group(name)
            closest_i = utils.find_closest_date(date, entries["DATE"].values)
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

        Returns games sorted in chronological order, with
        most recent games last.

        Args:
            name (str): name of player from nba api

        Returns:
            pd.DataFrame: dataframe of games played
        """
        
        if name in self.nba_stats.groups:
            return self.nba_stats.get_group(name)
        
        # Return empty df with the right columns if not there
        else:
            return self.nba_stats.get_group(list(self.nba_stats.groups)[0]).iloc[:0]


    def teamID_lookup(self, teamID: str) -> tuple:
        """
        Gets the manager and team names, given
        the Yahoo team id

        Args:
            teamID (str): Yahoo unique identifier for the team

        Returns:
            tuple: (manager name, team name)
        """

        return (self.fantasy_teams.at[teamID, 'manager'],
                self.fantasy_teams.at[teamID, 'teamName'])


    def games_in_week(self, nba_team: str, week: int,
                      upto: Union[str, datetime.datetime] = "") -> Union[int, tuple[int]]:
        """
        Returns the number of games in the given fantasy
        week for the given nba team

        Args:
            nba_team (str): NBA team 3 letter abbreviation
                            i.e., GSW
            week (int): Yahoo week in the season
            upto (str or datetime): instead return games
                                    before and after the given
                                    date (upto date is in after)

        Returns:
            int/tuple: number of games or (n_games before upto,
                                           n_games after and including upto)
        """
        # Check upto is in the right week
        if isinstance(upto, str):
            if upto != "":
                if not self.week_for_date(upto) == week:
                    raise(ValueError, "upto not in correct week")
        elif not self.week_for_date(upto) == week:
            raise(ValueError, "upto not in correct week")

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
            if date < upto_dt:
                counts[0] += self.games_per_day.at[date_str, nba_team]
            else:
                counts[1] += self.games_per_day.at[date_str, nba_team]

        if isinstance(upto, str):
            if upto == "":
                return counts[0]
            else:
                return tuple(counts)
        else:
            return tuple(counts)


    def matchup_score(self, week: int,
                      date: Union[str, datetime.datetime] = "") -> pd.DataFrame:
        """
        Returns the matchup score for a given week,
        on the morning of the specified date (if a date is given)
        i.e., before any games have been played

        Args:
            nba_team (str): NBA team 3 letter abbreviation
                            i.e., GSW
            week (int): Yahoo week in the season
            upto (str or datetime): instead return games
                                    before and after the given
                                    date (upto date is in before)

        Returns:
            pd.DataFrame: matchups for the week with scores
        """
        # Check date is in the right week
        if isinstance(date, str):
            # Convert from string to date
            if date != "":
                if not self.week_for_date(date) == week:
                    raise(ValueError, "date not in correct week")
                else:
                    date = datetime.datetime.strptime(date,
                                                      utils.DATE_SCHEMA)
            else:
                date = datetime.datetime.strptime(self.week_date_range(week)[1],
                                                  utils.DATE_SCHEMA)

        elif not self.week_for_date(date) == week:
            raise(ValueError, "date not in correct week")

        # Get all stats entries from the week where
        # people actually played
        stats = self.get_nba_stats(f"WHERE week = {week}")

        # Filter based on date
        to_keep = [datetime.datetime.strptime(x, utils.DATE_SCHEMA) < date
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
                if attempts > 0:
                    perc = (sched.at[team, made] / sched.at[team, attempts])
                else:
                    perc = 0
                sched.at[team, stat] = perc
                    

        return sched


if __name__ == "__main__":
    db = dbInterface("past_season_dbs/yahoo_fantasy_2022_23.sqlite")
