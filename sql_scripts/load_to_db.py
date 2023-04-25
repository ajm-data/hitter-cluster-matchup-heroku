import statsapi
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import psycopg2
from dotenv import load_dotenv
import os
from game_results import engine_connect_db_mlb
from ready_data import game_results, active_players
from sqlalchemy import create_engine

load_dotenv()

df = game_results()

ap = active_players()


game_results_df = game_results()

engine = engine_connect_db_mlb()
# engine.autocommit = True

cursor = engine.cursor()

sql1 = '''select * from completed_games;'''
cursor.execute(sql1)


game_results_df.to_sql('completed_games', con=engine, if_exists='replace',
          index=False)


sql1 = '''select * from completed_games;'''
cursor.execute(sql1)

for i in cursor.fetchall():
    print(i)

cursor.close()
engine.close()

conn_string = f"postgresql://{os.getenv('USER')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}:{os.getenv('PORT')}/{os.getenv('DATABASE')}"


db = create_engine(conn_string)
conn = db.connect()

df.to_sql('completed_games', con=conn, if_exists='replace',
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





conn_string = f"postgresql://{os.getenv('USER')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}:{os.getenv('PORT')}/{os.getenv('DATABASE')}"


db = create_engine(conn_string)
conn = db.connect()

ap.to_sql('active_players', con=conn, if_exists='replace',
          index=False)

conn = psycopg2.connect(conn_string
                        )

conn.autocommit = True
cursor = conn.cursor()
  
sql1 = '''select * from active_players;'''
cursor.execute(sql1)

for i in cursor.fetchall():
    print(i)
  
conn.commit()
conn.close()