import psycopg2
from sqlalchemy import create_engine
from get_cleaned_playermap import player_map_for_table

# Dataframe of all active players
df = player_map_for_table()
# # print("These are the columns", '[%s]' % ', '.join(map(str, df.columns)))
# print(df.columns)

hostname = 'localhost'
database = 'baseball'
username = 'postgres'
pwd = 'alec'
port_id = 5432

# # conn_string = "postgres://{postgres}:{alec}@{localhost}:{5432}/{playermap}"
conn_string = f"postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}"


db = create_engine(conn_string)
conn = db.connect()

df.to_sql('active_players', con=conn, if_exists='replace',
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









# # conn = None
# # cur = None

# try:
#     conn = psycopg2.connect(
#         host = hostname,
#         dbname = database,
#         user = username,
#         password = pwd,
#         port = port_id)
    
#     cur  = conn.cursor()
    
#     create_script = ''' CREATE TABLE IF NOT EXISTS active_players (
#                         MLBID INT NOT NULL PRIMARY KEY,
#                         IDPLAYER VARCHAR(11) NOT NULL,
#                         PLAYERNAME VARCHAR(50) NOT NULL,
#                         TEAM VARCHAR(5) NOT NULL,
#                         POS VARCHAR(5) NOT NULL,
#                         BATS VARCHAR(3) NOT NULL,
#                         THROWS VARCHAR(1) NOT NULL,
#                         ROTOWIREID INT NOT NULL,
#                         DRAFTKINGSNAME VARCHAR(50),
#                         FANDUELNAME VARCHAR(50),
#                         FANDUELID INT, 
#                         ALLPOS VARCHAR(10) NOT NULL)'''
    
#     cur.execute(create_script)

#     insert_script = 'INSERT INTO active_players (IDPLAYER, MLBID, PLAYERNAME, FIRSTNAME, LASTNAME, TEAM, POS, MLBNAME, BATS, THROWS, ALLPOS, ROTOWIREID, BREFID, FANDUELID, DRAFTKINGSNAME, ROTOWIRENAME, FANDUELNAME, DRAFTKINGSNAME)'
#     insert_values
#     conn.commit()
# except Exception as error:
#     print(error)

# finally:
#     if cur is not None:
#         cur.close()
#     if conn is not None:
#         conn.close()
    


# def get_connection():
#     return create_engine(
#         url="postgresql://{0}:{1}@{2}:{3}/{4}".format(
#             username, pwd, hostname, database
#         )
#     )
# get_connection()

# if __name__ == '__main__':
 
#     try:
#         # GET THE CONNECTION OBJECT (ENGINE) FOR THE DATABASE
#         engine = get_connection()
#         print(
#             f"Connection to the {hostname} for user {username} created successfully.")
#     except Exception as ex:
#         print("Connection could not be made due to the following error: \n", ex)