import statsapi
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import psycopg2


    # date handling
date_interval = timedelta(days=5)
x_current_date = date.today()
x_end_date = x_current_date + date_interval

current_date = "%s/%s/%s" % (x_current_date.month, x_current_date.day, x_current_date.year)
end_date = "%s/%s/%s" % (x_end_date.month, x_end_date.day, x_end_date.year)

##### Date of games #####
# sched = statsapi.schedule(start_date='4/1/2023',end_date=current_date)

# pddf = pd.DataFrame(sched, columns=['game_id', 'game_date','away_id', 'away_name', 'away_probable_pitcher', 'home_id', 'home_name', 'home_probable_pitcher',
#       'summary'])
#### Summary of Game Results ##################

#Postgres
# #establishing the connection
conn = psycopg2.connect(
   database="baseball", user='postgres', password='alec', host='127.0.0.1', port= '5432'
)
conn.autocommit = True

# #Creating a cursor object using the cursor() method
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS completed_games")

#Preparing query to create a database
create_script = ''' CREATE TABLE completed_games (
        game_id INT NOT NULL PRIMARY KEY,
        game_date DATE NOT NULL,
        away_id INT NOT NULL,
        away_name VARCHAR(30) NOT NULL,
        away_probable_pitcher VARCHAR(30) NOT NULL,
        home_id INT NOT NULL,
        home_name VARCHAR(30) NOT NULL,
        home_probable_pitcher VARCHAR(30) NOT NULL,
        summary VARCHAR(250) NOT NULL
        )''';

#Creating a table
cursor.execute(create_script)
print("Table created successfully........")

#Closing the connection
conn.close()

