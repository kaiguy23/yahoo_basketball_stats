import pandas as pd
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
        self.nba_lookup = None
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
    
    def get_nba_rosters(self, filter_statement = ""):
        """
        
        db.get_fantasy_schedule("WHERE manager LIKE 'Eli'")

        Args:
            filter_statement (str, optional): _description_. Defaults to "".
            season (_type_, optional): _description_. Defaults to utils.DEFAULT_SEASON.

        Returns:
            _type_: _description_
        """
        table_name = f"NBA_ROSTERS_{self.season}"
        query = dbInterface.build_select_statement(table_name, filter_statement)
        df = pd.read_sql_query(query, self.con)
        return df
    
    def get_fantasy_teams(self):
        table_name = "CURRENT_FANTASY_TEAMS"
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)
    

    def week_date_range(self, week):

        # Build the weeks dataframe if it hasn't been built yet
        if self.weeks is None:
            fantasy_schedule = self.get_fantasy_schedule()
            self.weeks = fantasy_schedule[fantasy_schedule.teamID == fantasy_schedule.teamID.iloc[-1]][['week', 'startDate', 'endDate']]
            self.weeks.index = self.weeks.week
        
        # Return the start and end dates
        return (self.weeks.at[week, 'startDate'], self.weeks.at[week, 'endDate'])
    
    # TODO: Maybe include lookup over stats to see who they were playing for?
    def player_affiliation(self, name, date):

        if isinstance(date, str):
            date = datetime.datetime.strptime(date, utils.DATE_SCHEMA)
        date_str = date.strftime(utils.DATE_SCHEMA)



        # Generate lookups if they haven't been made yet
        if self.fantasy_lookup is None:
            self.fantasy_lookup = self.get_fantasy_rosters().groupby("name")

        if self.nba_lookup is None:
            self.nba_lookup = self.get_nba_rosters().groupby("PLAYER_NAME")
        
        # Find which NBA team the player was on for the specified date
        nba_entries = self.nba_lookup.get_group(name).sort_values("START_DATE", ascending=True)
        nba_team = ""
        for i, row in nba_entries.iterrows():
            if row['END_DATE'] != "":
                start_date = datetime.datetime.strptime(row["START_DATE"], utils.DATE_SCHEMA)
                end_date = datetime.datetime.strptime(row["END_DATE"], utils.DATE_SCHEMA)
                if start_date <= date < end_date:
                    nba_team = row["TEAM_ABBREVIATION"]
            else:
                nba_team = row["TEAM_ABBREVIATION"]

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

        


            
        

## TODO: Make a lookup player stats method to get stats for one individual player

## TODO: Make games played this week so far function using unique values of (date, team) in player_stats

## TODO: Make a function that gets the matchup score on any given day by summing player stats