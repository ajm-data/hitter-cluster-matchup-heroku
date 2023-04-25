import statsapi
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import psycopg2
from sqlalchemy import create_engine


 #-----   # Jank date handling -----#
date_interval = timedelta(days=5)
x_current_date = date.today()
x_end_date = x_current_date + date_interval

current_date = "%s/%s/%s" % (x_current_date.month, x_current_date.day, x_current_date.year)
end_date = "%s/%s/%s" % (x_end_date.month, x_end_date.day, x_end_date.year)
#---------- #################------------#

# Need to worry think about duplicate values, how to handle 
def game_results():
    
    sched = statsapi.schedule(start_date='4/21/2023',end_date='4/24/2023')
    
    cols = ['game_id', 'game_date','away_id', 'away_name', 'away_probable_pitcher', 
            'home_id', 'home_name', 'home_probable_pitcher', 'summary']
    
    game_results_df = pd.DataFrame(sched, columns= cols)

    return game_results_df

def active_players():

    player_id_map_csv = pd.read_csv('SFBB Player ID Map - PLAYERIDMAP.csv')
    player_id_map_df = pd.DataFrame(player_id_map_csv)
    player_id_map_df.columns = [x.lower() for x in player_id_map_df.columns]
    
    cols = ['idplayer', 'mlbid', 'playername', 'firstname', 'lastname', 
        'team', 'pos', 'mlbname', 'bats', 'throws', 'allpos', 
        'rotowireid', 'brefid', 'fanduelid', 'rotowirename', 
        'fanduelname', 'draftkingsname']

    # Filter for active players
    active_player_map = player_id_map_df[cols].loc[ (player_id_map_df['active'] == 'Y')]
    
    return active_player_map


