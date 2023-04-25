import statsapi
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import psycopg2
from sqlalchemy import create_engine


    # date handling
# date_interval = timedelta(days=5)
# x_current_date = date.today()
# x_end_date = x_current_date + date_interval

# current_date = "%s/%s/%s" % (x_current_date.month, x_current_date.day, x_current_date.year)
# end_date = "%s/%s/%s" % (x_end_date.month, x_end_date.day, x_end_date.year)

#### Date of games #####
sched = statsapi.schedule(start_date='4/1/2023',end_date='4/24/2023')

game_results_df = pd.DataFrame(sched, columns=['game_id', 'game_date','away_id', 'away_name', 'away_probable_pitcher', 'home_id', 'home_name', 'home_probable_pitcher',
    'summary'])
### Summary of Game Results ##################

hostname = 'localhost'
database = 'baseball'
username = 'postgres'
pwd = 'alec'
port_id = 5432

conn_string = f"postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}"

conn_string

### Date of games #####
# sched = statsapi.schedule(start_date='4/1/2023',end_date=current_date)

# game_results_df = pd.DataFrame(sched, columns=['game_id', 'game_date','away_id', 'away_name', 'away_probable_pitcher', 'home_id', 'home_name', 'home_probable_pitcher',
#     'summary'])
# ### Summary of Game Results ##################

# hostname = 'localhost'
# database = 'game_data'
# username = 'postgres'
# pwd = 'alec'
# port_id = 5432
# conn_string = f"postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}"

db = create_engine(conn_string)
conn = db.connect()

game_results_df.to_sql('completed_games', con=conn, if_exists='replace',
          index=False)

conn = psycopg2.connect(conn_string
                        )
conn.autocommit = True
cursor = conn.cursor()
  
sql1 = '''select * from completed_games;'''
cursor.execute(sql1)

for i in cursor.fetchall():
    print(i)
  
conn.commit()
conn.close()