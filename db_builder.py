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
#   6) (title: GAMES_PER_WEEK_{SEASON})
#       - Shows the number of games per week for each NBA team
#   
#   7) (title: NBA_ROSTERS_{SEASON}) NBA rosters for the selected season. Shows the date
#                                    each entry was updated, so you can see when players
#                                    switched teams
####


## TODO: Make a logger?

import sqlite3
import datetime
import time
import pandas as pd
import numpy as np
from pathlib import Path

import utils
from nba_api.stats.endpoints import playergamelogs, scoreboard, commonteamroster
from db_interface import dbInterface


class dbBuilder:

    def __init__(self, db_file, oauth_file = 'yahoo_oauth.json', season = utils.DEFAULT_SEASON, debug=False):
        
        self.oauth_file = oauth_file 
        self.sc, self.gm, self.lg = utils.refresh_oauth_file(self.oauth_file)
        self.db_file = db_file
        self.con = sqlite3.connect(f)
        self.cur = self.con.cursor()
        self.season = season
        self.debug = debug
    
    def __del__(self):
        self.con.close()

    
    def refresh_oath(self):
        self.sc, self.gm, self.lg = utils.refresh_oauth_file(self.oauth_file)

    
    def table_names(self):

        return self.cur.execute( f"SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    def check_table_exists(self, table_name):
        
        tables = self.cur.execute(
        f"""SELECT name FROM sqlite_master WHERE type='table'
        AND name='{table_name}'; """).fetchall()
        
        if tables == []:
            return False
        else:
            return True
    
    def delete_table(self, table_name):
        self.cur.execute(f"DROP TABLE {table_name}")

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
                        start_week = i-1
                        break
                start_week = current_week
            else:
                start_week = max_with_schedule

        # Get matchups from each week
        for week in range(start_week, end_week+1): 

            if self.debug:
                print("Updating Fantasy Schedule for Week", week)

            # Delete any entries in the db
            if table_exists:
                self.cur.execute(f"DELETE FROM {table_name} WHERE week LIKE '{week}'")
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


            df = utils.extract_matchup_scores(self.lg, week, nba_cols=False)
            df['week'] = week
            df['startDate'] = start_day
            df['endDate'] = end_day
            df.to_sql(table_name, self.con, if_exists="append",index=False)
            self.con.commit()


        return


    def update_player_stats(self):
        """Must be run AFTER updating fantasy rosters
        """
        # Start and end days to get stats for
        start_day = self.lg.week_date_range(1)[0]
        end_day = self.lg.week_date_range(self.lg.end_week())[1]

        # Get the fantasy rosters
        db_reader = dbInterface(f)
        fantasy_rosters = db_reader.get_fantasy_rosters()
        fantasy_roster_dates = (fantasy_rosters['date'].min(), fantasy_rosters['date'].max())
        fantasy_rosters = fantasy_rosters.groupby("date")

        # Figure out how far existing stats go
        table_name = f"PLAYER_STATS_{self.season}"
        table_exists = self.check_table_exists(table_name=table_name)

        if self.debug:
            print("Updating Player Stats")

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
            # There may be some days without nba games
            if date_str not in stats.indices:
                continue

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
    
    def update_nba_rosters(self):
        # MUST RUN AFTER NBA SCHEDULE

        table_name = f"NBA_ROSTERS_{self.season}"
        table_exists = self.check_table_exists(table_name=table_name)

        # Get all the NBA teams and their NBA TEAM_ID
        db_reader = dbInterface(f)
        nba_schedule = db_reader.get_nba_schedule()
        nba_by_team = nba_schedule.groupby("TEAM_ABBREVIATION")

        # If we've made rosters before
        # delete any partial entries 
        # with this date
        if table_exists:
            self.cur.execute(f"DELETE FROM {table_name} WHERE\
                                DATE LIKE '{utils.TODAY_STR}'")
            self.con.commit()


        all_rosters = []
        for team in nba_by_team.indices:

            if self.debug:
                print("Getting Rosters for", team)
            
            # Get the current NBA Rosters
            entry = nba_by_team.get_group(team).iloc[0]
            roster = commonteamroster.CommonTeamRoster(entry['TEAM_ID'], self.season).get_data_frames()[0]
            roster.rename(columns = {"PLAYER": "PLAYER_NAME",
                                     "TeamID": "TEAM_ID"}, inplace=True)
            roster["TEAM_ABBREVIATION"] = team
            roster["DATE"] = utils.TODAY_STR

            all_rosters.append(roster)
            
            # So we don't do too many requests
            time.sleep(1)


        all_rosters = pd.concat(all_rosters)
        all_rosters.to_sql(table_name, self.con, if_exists="replace",index=True)
        self.con.commit()
            

    
    def update_num_games_per_week(self):
        # MUST BE RUN AFTER UPDATE NBA SCHEDULE
        # AND UPDATE FANTASY SCHEDULE 
        
        table_name = f"GAMES_PER_WEEK_{self.season}"

        # Get the fantasy and nba schedules
        db_reader = dbInterface(f)
        fantasy_schedule = db_reader.get_fantasy_schedule()
        nba_schedule = db_reader.get_nba_schedule()
        nba_by_day = nba_schedule.groupby("GAME_DATE")

        # Only process weeks 
        # that are present in both the fantasy schedule
        # And the NBA schedule
        start_week = fantasy_schedule['week'].min()
        end_week = fantasy_schedule['week'].max()

        all_weeks = []
        for week in range(start_week, end_week+1):

            if self.debug:
                print("Getting Num Games in Week", week)

            start_day_str, end_day_str = db_reader.week_date_range(week)
            start_day = datetime.datetime.strptime(start_day_str, utils.DATE_SCHEMA)
            end_day = datetime.datetime.strptime(end_day_str, utils.DATE_SCHEMA)
            n_games = {'week': week, 'startDate': start_day_str, 'endDate': end_day_str}

            for date in pd.date_range(start_day, end_day, freq='D'):
                date_str = date.strftime(utils.DATE_SCHEMA)
                
                # Some dates don't have NBA games
                if date_str not in nba_by_day.indices:
                    continue

                games = nba_by_day.get_group(date_str)
                # Count each team that played today
                for i, row in games.iterrows():
                    team = row['TEAM_ABBREVIATION']
                    if team in n_games:
                        n_games[team] += 1
                    else:
                        n_games[team] = 1
            
            all_weeks.append(n_games)

        all_weeks = pd.DataFrame(all_weeks)
        all_weeks.set_index("week", inplace=True)
        all_weeks.to_sql(table_name, self.con, if_exists="replace",index=True)
        self.con.commit()

        return
    
    # NOTE: SLOW because we have to do an API request for each day
    def update_nba_schedule(self):
        
        start_day = self.lg.week_date_range(1)[0]
        end_day = self.lg.week_date_range(self.lg.end_week())[1]

        # Find the last day we have games for, and find the last 
        # day we have game stats for (i.e. PTS aren't None)
        # set start and end day accordingly
        table_name = f"NBA_SCHEDULE_{self.season}"
        table_exists = self.check_table_exists(table_name=table_name)

        if table_exists:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)
            max_date = df['GAME_DATE'].max()

            # We have schedule but may need to update stats
            if max_date == end_day:
                end_day = utils.TODAY
                grouped = df.groupby("GAME_DATE")
                for i in sorted(grouped.indices):
                    for_day = grouped.get_group(i)
                    if all([np.isnan(x) for x in for_day['PTS'].values]):
                        start_day = i-1
                        break

            # We need to update the schedule through end of season
            else:
                start_day = max_date

        for date in pd.date_range(start_day, end_day, freq='D'):


            date_str = date.strftime(utils.DATE_SCHEMA)

            if self.debug:
                print("Getting Schedule for", date_str)

            # Delete any entries in the db
            if table_exists:
                self.cur.execute(f"DELETE FROM {table_name} WHERE GAME_DATE LIKE '{date_str}'")
                self.con.commit()

            sb = scoreboard.Scoreboard(game_date=date).get_data_frames()[1]
            sb.rename(columns={"GAME_DATE_EST":"GAME_DATE"}, inplace=True)
            sb.rename(columns=utils.NBA_TO_YAHOO_STATS, inplace=True)
            sb['GAME_DATE'] = [x[:10] for x in sb['GAME_DATE']]

            
            # Filter out for all-star games
            sb = sb[sb['TEAM_ABBREVIATION'].isin(utils.NBA_TEAMS_SET)]

            sb.to_sql(table_name, self.con, if_exists="append",index=False)
            self.con.commit()

            # So we don't do too many requests
            time.sleep(1)


            
        return

    def update_fantasy_teams(self):
        team_df = utils.get_team_ids(self.sc,self.lg).drop('teamObject', axis=1)
        team_df.to_sql("CURRENT_FANTASY_TEAMS", self.con, if_exists="replace",index=False)
        return 


    def update_fantasy_rosters(self, pace = False, limit_per_hour=300):

        if self.debug:
            print("Updating Fantasy Rosters")

        start_day = self.lg.week_date_range(1)[0]
        end_day = self.lg.week_date_range(self.lg.end_week())[1]

        table_name = f"FANTASY_ROSTERS_{self.season}"
        table_exists = self.check_table_exists(table_name=table_name)

        


        # If the table exists, find the most recent day we have rosters for
        if table_exists:
            start_date_str = self.cur.execute(f"SELECT MAX(date) FROM {table_name}").fetchone()[0]
            if not start_date_str is None:
                start_day = datetime.datetime.strptime(start_date_str, utils.DATE_SCHEMA)

                # Remove the most recent day, as we'll be overwriting it to make sure
                # we got final rosters for the day
                self.cur.execute(f"DELETE FROM {table_name} WHERE date LIKE '{start_date_str}'")
                self.con.commit()

        team_df = utils.get_team_ids(self.sc,self.lg)
        query_count = 3
        for date in pd.date_range(start_day, end_day, freq='D'):
            all_rosters = []
            for i in range(team_df.shape[0]):
                
                row = team_df.iloc[i]

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

                if self.debug:
                    print("Getting Roster for", current_roster['teamName'].iloc[0], current_roster['date'].iloc[0])

                query_count += 1
                if pace:
                    if query_count > limit_per_hour-20:
                        # Sleep an hour so we don't run out of Yahoo API requests
                        print("Sleepy Time...")
                        time.sleep(60*60)

                        # Refresh the access
                        self.refresh_oath()
                        team_df = utils.get_team_ids(self.sc,self.lg)
                        query_count = 0

                    

            # Save results from each day in case we go over the query limit
            all_rosters = pd.concat(all_rosters)
            all_rosters['eligible_positions'] = [str(x) for x in all_rosters['eligible_positions'].values]
            all_rosters.to_sql(table_name, self.con, if_exists="append",index=False)
            self.con.commit()

        return
        
        
    def update_db(self):
        self.update_fantasy_teams()
        self.update_fantasy_schedule()
        self.update_fantasy_rosters(pace=False)
        self.update_player_stats()
        self.update_nba_schedule()
        self.update_num_games_per_week()
        self.update_nba_rosters()
            


if __name__ == "__main__":

    f = "yahoo_save.sqlite"
    # builder = dbBuilder(f, debug=True)
    

    con = sqlite3.connect(f)
    cur = con.cursor()


    db = dbInterface(f)















