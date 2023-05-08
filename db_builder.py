####
# Builds a databse for use in fantasy basketball analysis.
# Season is of format 2022-23
# Tables are 
#   1) (title: PLAYER_STATS_{SEASON}) Each player's stats for each game in a season
#       - This table also shows which fantasy team they were rostered on (if any) at game time, and what their position was
#
#   2) (title: NBA_SCHEDULE_{SEASON}) NBA schedule for a given season, i.e. games and which team is playing 
#       - 
#
#   3) (title: FANTASY_SCEHEDULE_{SEASON}) Yahoo fantasy schedule for a given season 
#       - Updates with scores as they come in
#       - Shows the week number and the dates associated with that week
#
#   4) (title: CURRENT_FANTASY_TEAMS) Current team names/id's
#
#   5) (title: FANTASY_ROSTERS_{SEASON}) Fantasy rosters for each day through  
#
#   6) (title: NUM_GAMES_PER_WEEK_{SEASON})
#       - Shows the number of games per week for each individual player, updates
#         when they switch teams in the NBA.
#
####

import sqlite3
import datetime
import time
import sqlalchemy as sqa
import pandas as pd
from pathlib import Path

import utils
from nba_api.stats.endpoints import playergamelogs
from db_interface import dbInterface


class dbBuilder:

    def __init__(self, db_file, oauthFile = 'yahoo_oauth.json', season = utils.DEFAULT_SEASON):
        
        self.sc, self.gm, self.lg = utils.refresh_oauth_file(oauthFile)
        self.db_file = db_file
        self.con = sqlite3.connect(f)
        self.cur = self.con.cursor()
        self.season = season
    
    def __del__(self):
        self.con.close()

    def check_table_exists(self, table_name):
        
        tables = self.cur.execute(
        f"""SELECT name FROM sqlite_master WHERE type='table'
        AND name='{table_name}'; """).fetchall()
        
        if tables == []:
            return False
        else:
            return True


    def update_fantasy_schedule(self):
        
        # Get information about the current time in the league
        start_week = 1
        end_week = self.lg.end_week()
        current_week = self.lg.current_week()


        # Check what the most recent fantasy week in the database is
        table_name = f"FANTASY_SCHEDULE_{self.season}"
        table_exists = self.check_table_exists(table_name=table_name)
        if table_exists:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)
            max_with_schedule = df['week'].max()

            # If we already have a schedule through the end of the year
            # Focus on updating any stats that may need updating
            if max_with_schedule == end_week:
                end_week = current_week
                grouped = df.groupby("week")
                for i in sorted(grouped.indices):
                    for_week = grouped.get_group(i)
                    if all(for_week['PTS'] == 0.0):
                        start_week = week-1
                        break
                start_week = current_week
            else:
                start_week = max_with_schedule

        # Get matchups from each week
        for week in range(start_week, end_week+1): 

            # Delete any entries in the db
            if table_exists:
                cur.execute(f"DELETE FROM {table_name} WHERE week LIKE '{week}'")
                self.con.commit()

            # Try to get the date range for the given week, may error if 
            # it's too far in the future
            try:
                start_day, end_day = self.lg.week_date_range(week)
                start_day = start_day.strftime(utils.DATE_SCHEMA)
                end_day = end_day.strftime(utils.DATE_SCHEMA)

            except:
                start_day = ""
                end_day = ""


            df = utils.extract_matchup_scores(self.lg, week)
            df['week'] = week
            df['startDate'] = start_day
            df['endDate'] = end_day
            df.to_sql(table_name, self.con, if_exists="append",index=False)
            self.con.commit()


        return


    def update_player_stats(self):
        """Must be run AFTER updating fantasy rosters
        """
        # TODO: CHANGE START DAY WHEN DONE DEBUGGING
        # Start and end days to get stats for
        start_day = self.lg.week_date_range(20)[0]
        end_day = self.lg.week_date_range(self.lg.end_week())[1]

        # Get the fantasy rosters
        db_reader = dbInterface(f)
        fantasy_rosters = db_reader.get_fantasy_rosters()
        fantasy_roster_dates = (fantasy_rosters['date'].min(), fantasy_rosters['date'].max())
        fantasy_rosters = fantasy_rosters.groupby("date")

        # Figure out how far existing stats go
        table_name = f"PLAYER_STATS_{self.season}"
        table_exists = self.check_table_exists(table_name=table_name)

        team_df = utils.get_team_ids(self.sc,self.lg)

        # If the table exists, find the most recent day we have stats
        if table_exists:
            start_date_str = self.cur.execute(f"SELECT MAX(GAME_DATE) FROM {table_name}").fetchone()[0]
            if not start_date_str is None:
                start_day = datetime.datetime.strptime(start_date_str, utils.DATE_SCHEMA)

                # Remove the most recent day, as we'll be overwriting it to make sure
                # we got final rosters for the day
                self.cur.execute(f"DELETE FROM {table_name} WHERE GAME_DATE LIKE '{start_date_str}'")
                self.con.commit()


        # Get the full nba stats for the season in question
        stats = playergamelogs.PlayerGameLogs(season_nullable=self.season.replace("_","-")
                ).get_data_frames()[0]
        stats.rename(columns=utils.NBA_TO_YAHOO_STATS, inplace=True)
        stats['GAME_DATE'] = [x[:10] for x in stats['GAME_DATE']]
        stats = stats.groupby("GAME_DATE")
        
        # Loop through days we want stats for
        all_entries = []
        for date in pd.date_range(start_day, end_day, freq='D'):

            date_str = date.strftime(utils.DATE_SCHEMA)  
            
            # Get stats for the day
            nba_stats = stats.get_group(date_str)
            fantasy = fantasy_rosters.get_group(date_str).groupby("name")
            
            # Append fantasy roster columns to the NBA API Columns
            new_df = nba_stats.copy(deep=True)
            cols_to_add = ["status", "position_type", "eligible_positions",
                           "selected_position","teamID", "manager", "teamName"]
            for col in cols_to_add:
                new_df[col] = ""
            

            # Add fantasy information for players that are there
            for i, row in nba_stats.iterrows():
                if row['PLAYER_NAME'] in fantasy.indices:
                    f_stats = fantasy.get_group(row['PLAYER_NAME']).iloc[0]
                    for col in cols_to_add:
                        new_df.at[i, col] = f_stats[col]
            
            all_entries.append(new_df)

        all_entries = pd.concat(all_entries)
        all_entries.to_sql(table_name, self.con, if_exists="append",index=False)
        self.con.commit()
        return 
    
    # TODO: Add this
    def update_num_games_per_week(self):
        return
    
    # TODO: Add this
    # NOTE: SLOW
    def update_nba_schedule(self):

        return

    def update_fantasy_teams(self):
        team_df = utils.get_team_ids(self.sc,self.lg).drop('teamObject', axis=1)
        team_df.to_sql("CURRENT_FANTASY_TEAMS", self.con, if_exists="replace",index=False)
        return 


    def update_fantasy_rosters(self, pace = False, limit_per_hour=350):
        
        # TODO: CHANGE START DAY WHEN DONE DEBUGGING
        start_day = self.lg.week_date_range(20)[0]
        end_day = self.lg.week_date_range(self.lg.end_week())[1]

        table_name = f"FANTASY_ROSTERS_{self.season}"
        table_exists = self.check_table_exists(table_name=table_name)

        team_df = utils.get_team_ids(self.sc,self.lg)

        # If the table exists, find the most recent day we have rosters for
        if table_exists:
            start_date_str = self.cur.execute(f"SELECT MAX(date) FROM {table_name}").fetchone()[0]
            if not start_date_str is None:
                start_day = datetime.datetime.strptime(start_date_str, utils.DATE_SCHEMA)

                # Remove the most recent day, as we'll be overwriting it to make sure
                # we got final rosters for the day
                self.cur.execute(f"DELETE FROM {table_name} WHERE date LIKE '{start_date_str}'")
                self.con.commit()

        query_count = 0
        for date in pd.date_range(start_day, end_day, freq='D'):
            all_rosters = []
            for i, row in team_df.iterrows():
                current_roster = pd.DataFrame(row['teamObject'].roster(day = date))
                
                # Add date information
                current_roster['date'] = date.strftime(utils.DATE_SCHEMA)

                # Add team information
                for col in ["teamID", "manager", "teamName"]:
                    current_roster[col] = row[col]

                # Add nba api name information
                current_roster['yahoo_name'] = current_roster['name'].copy()
                current_roster['name'] = [utils.yahoo_to_nba_name(x) for x in current_roster['yahoo_name'].values]

                # Save the results
                all_rosters.append(current_roster)

                query_count += 1
                if pace:
                    if query_count > limit_per_hour-20:
                        # Sleep an hour so we don't run out of Yahoo API requests
                        time.sleep(60*60)
                        query_count = 0

                # print(current_roster['teamName'].iloc[0], current_roster['date'].iloc[0])
                    

            # Save results from each day in case we go over the query limit
            all_rosters = pd.concat(all_rosters)
            all_rosters['eligible_positions'] = [str(x) for x in all_rosters['eligible_positions'].values]
            all_rosters.to_sql(table_name, self.con, if_exists="append",index=False)
            self.con.commit()
            return
            
            


if __name__ == "__main__":

    f = "yahoo_fantasy.sqlite"
    builder = dbBuilder(f)
    # builder.update_fantasy_teams()
    # builder.update_fantasy_schedule()
    # builder.update_fantasy_rosters()
    builder.update_player_stats()

    con = sqlite3.connect(f)
    cur = con.cursor()

    db = dbInterface(f)















