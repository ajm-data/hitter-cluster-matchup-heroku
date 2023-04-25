# hostname = 'localhost'
# database = 'demo'
# username = 'postgres'
# pwd = 'alec'
# port_id = 5432

import psycopg2
from get_cleaned_playermap import player_map_for_table


player_map_dataframe = player_map_for_table()

# #establishing the connection
conn = psycopg2.connect(
   database="baseball", user='postgres', password='alec', host='127.0.0.1', port= '5432'
)
conn.autocommit = True

# #Creating a cursor object using the cursor() method
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS active_players")

#Preparing query to create a database
# should turn this into a function to call for other table creations 
create_script = ''' CREATE TABLE active_players (
        MLBID INT NOT NULL PRIMARY KEY,
        IDPLAYER VARCHAR(11) NOT NULL,
        PLAYERNAME VARCHAR(50) NOT NULL,
        TEAM VARCHAR(5) NOT NULL,
        POS VARCHAR(5) NOT NULL,
        BATS VARCHAR(3) NOT NULL,
        THROWS VARCHAR(1) NOT NULL,
        ROTOWIREID INT NOT NULL,
        DRAFTKINGSNAME VARCHAR(50),
        FANDUELNAME VARCHAR(50),
        FANDUELID INT,
        ALLPOS VARCHAR(10) NOT NULL

 )''';

#Creating a table
cursor.execute(create_script)
print("Table created successfully........")

#Closing the connection
conn.close()

# from sqlalchemy import create_engine
# engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')
# df.to_sql('table_name', engine)