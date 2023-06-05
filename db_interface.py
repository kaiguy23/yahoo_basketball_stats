import pandas as pd
import numpy as np
import sqlite3
import datetime

import utils


class dbInterface:

    def __init__(self, f):

        # Open in read only mode
        self.con = sqlite3.connect(f, uri=True)
        self.cur = self.con.cursor()
        self.weeks = None
        self.fantasy_lookup = None
        self.nba_stats = None
        self.nba_rosters = None
        self.fantasy_teams = None
        self.games_per_week = None

        
    def __del__(self):
        self.con.close()

    @staticmethod
    def build_select_statement(table_name, filter_statement=""):
        query = f"SELECT * FROM {table_name}"
        if filter_statement != "":
            query += " " + filter_statement
        return query
    
    def table_names(self):

        return self.cur.execute( f"SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    def get_fantasy_schedule(self, filter_statement = "", season=utils.DEFAULT_SEASON):
        """
        
        db.get_fantasy_schedule("WHERE manager LIKE 'Eli'")

        Args:
            filter_statement (str, optional): _description_. Defaults to "".
            season (_type_, optional): _description_. Defaults to utils.DEFAULT_SEASON.

        Returns:
            _type_: _description_
        """
        table_name = f"FANTASY_SCHEDULE_{season}"
        query = dbInterface.build_select_statement(table_name, filter_statement)
        return pd.read_sql_query(query, self.con)
    
    def get_nba_schedule(self, filter_statement = "", season=utils.DEFAULT_SEASON):
        """
        
        db.get_fantasy_schedule("WHERE manager LIKE 'Eli'")

        Args:
            filter_statement (str, optional): _description_. Defaults to "".
            season (_type_, optional): _description_. Defaults to utils.DEFAULT_SEASON.

        Returns:
            _type_: _description_
        """
        table_name = f"NBA_SCHEDULE_{season}"
        query = dbInterface.build_select_statement(table_name, filter_statement)
        return pd.read_sql_query(query, self.con)
    
    def get_fantasy_rosters(self, filter_statement = "", season=utils.DEFAULT_SEASON):
        """
        
        db.get_fantasy_schedule("WHERE manager LIKE 'Eli'")

        Args:
            filter_statement (str, optional): _description_. Defaults to "".
            season (_type_, optional): _description_. Defaults to utils.DEFAULT_SEASON.

        Returns:
            _type_: _description_
        """
        table_name = f"FANTASY_ROSTERS_{season}"
        query = dbInterface.build_select_statement(table_name, filter_statement)
        return pd.read_sql_query(query, self.con)
    
    def get_player_stats(self, filter_statement = "", season=utils.DEFAULT_SEASON):
        """
        
        db.get_fantasy_schedule("WHERE manager LIKE 'Eli'")

        Args:
            filter_statement (str, optional): _description_. Defaults to "".
            season (_type_, optional): _description_. Defaults to utils.DEFAULT_SEASON.

        Returns:
            _type_: _description_
        """
        table_name = f"PLAYER_STATS_{season}"
        query = dbInterface.build_select_statement(table_name, filter_statement)
        return pd.read_sql_query(query, self.con)
    
    def get_games_per_week(self, filter_statement = "", season=utils.DEFAULT_SEASON):
        """
        
        db.get_fantasy_schedule("WHERE manager LIKE 'Eli'")

        Args:
            filter_statement (str, optional): _description_. Defaults to "".
            season (_type_, optional): _description_. Defaults to utils.DEFAULT_SEASON.

        Returns:
            _type_: _description_
        """
        table_name = f"GAMES_PER_WEEK_{season}"
        query = dbInterface.build_select_statement(table_name, filter_statement)
        df = pd.read_sql_query(query, self.con)
        df.index = df['week']
        return df
    
    def get_nba_rosters(self, filter_statement = "", season=utils.DEFAULT_SEASON):
        """
        
        db.get_fantasy_schedule("WHERE manager LIKE 'Eli'")

        Args:
            filter_statement (str, optional): _description_. Defaults to "".
            season (_type_, optional): _description_. Defaults to utils.DEFAULT_SEASON.

        Returns:
            _type_: _description_
        """
        table_name = f"NBA_ROSTERS_{season}"
        query = dbInterface.build_select_statement(table_name, filter_statement)
        df = pd.read_sql_query(query, self.con)
        return df
    
    def get_fantasy_teams(self):
        table_name = "CURRENT_FANTASY_TEAMS"
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)
    

    def week_date_range(self, week, season=utils.DEFAULT_SEASON):

        # Build the weeks dataframe if it hasn't been built yet
        if self.weeks is None:
            fantasy_schedule = self.get_fantasy_schedule(season=season)
            self.weeks = fantasy_schedule[fantasy_schedule.teamID ==
                                          fantasy_schedule.teamID.iloc[-1]][['week', 'startDate', 'endDate']]
            self.weeks.index = self.weeks.week
        
        # Return the start and end dates
        return (self.weeks.at[week, 'startDate'], self.weeks.at[week, 'endDate'])
    
    def player_affiliation(self, name, date):
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
    
    def teamID_lookup(self, teamID):

        if self.fantasy_teams is None:
            self.fantasy_teams = self.get_fantasy_teams()
            self.fantasy_teams.index = self.fantasy_teams.teamID
        
        return (self.fantasy_teams.at[teamID, 'manager'], 
                self.fantasy_teams.at[teamID, 'teamName'])
    
    def games_in_week(self, nba_team, week):

        if self.games_per_week is None:
            self.games_per_week = self.get_games_per_week()

        return self.games_per_week.at[week, nba_team]



# Helper function for finding the closest date
def find_closest_date(d, dates):
    """
    Finds the index of the closest date in 
    dates to the date d. If d/dates
    are strings, assumes they are in the default
    date format.

    Args:
        d (str or datetime): _description_
        dates (list of str or datetime): _description_
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



    


## TODO: Make season an input for the constructor
        

## TODO: Make a lookup player stats method to get stats for one individual player

## TODO: Make games played this week so far function using unique values of (date, team) in player_stats

## TODO: Make a function that gets the matchup score on any given day by summing player stats