import pandas as pd
import sqlite3

import utils


class dbInterface:

    def __init__(self, f):

        # Open in read only mode
        self.con = sqlite3.connect(f, uri=True)
        self.cur = self.con.cursor()
        
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
    
    def get_fantasy_teams(self):
        table_name = "CURRENT_FANTASY_TEAMS"
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)
    