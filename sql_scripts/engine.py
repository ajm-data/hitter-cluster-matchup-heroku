import statsapi
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def engine_connect_db_mlb():
    engine = psycopg2.connect(
        database=os.getenv("DATABASE"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"),
        port=os.getenv("PORT"))   
    
    return engine

def create_table_db_mlb():
    
    engine = engine_connect_db_mlb()
    engine.autocommit = True

    # #Creating a cursor object using the cursor() method
    cursor = engine.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS completed_games")

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
    engine.close()

