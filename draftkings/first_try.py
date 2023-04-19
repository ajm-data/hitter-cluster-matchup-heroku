import requests
from datetime import date, timedelta
import statsapi

# x = date.today()
# print(x)
# y = timedelta(days=5)
# print(x+y)
# end = x+y
# print(type(x))





# today = date.today()
# todays_date = "%s/%s/%s" % (end.month, end.day, end.year)
# print(todays_date)

date_interval = timedelta(days=5)
x_current_date = date.today()
x_end_date = x_current_date + date_interval

current_date = "%s/%s/%s" % (x_current_date.month, x_current_date.day, x_current_date.year)
end_date = "%s/%s/%s" % (x_end_date.month, x_end_date.day, x_end_date.year)

ending_date = date.today() + timedelta(days=5)

# url = 'https://www.draftkings.com/lobby/getcontests?sport=mlb'

# response = requests.get(url)

# print(len(response.text))
import pandas as pd

all_pitchers = pd.read_csv('Named_Clustered_Metric.csv')
all_pitchers_df = pd.DataFrame(all_pitchers)
all_pitchers_df['full_name'] = all_pitchers_df['first_name'] + ' ' + all_pitchers_df['last_name']
all_pitchers_df['full_name'] = all_pitchers_df['full_name'].str.strip()

all_pitchers_df = all_pitchers_df[['Cluster', 'full_name']]

# future_date = todays_date + datetime.timedelta(days=5)

sched = statsapi.schedule(start_date='4/14/2023',end_date='4/18/2023',team=108)
probable_pitcher = []
angels_pitcher = []
opposing_pitcher = []
game_date = []
opposing_pitcher_cluster = []
opposing_team = []

for i in range(0, len(sched)):
    if (sched[i]['home_probable_pitcher']=='') or (sched[i]['away_probable_pitcher']==''):
        break
    else:
        game_date.append(sched[i]['game_date'])
        if sched[i]['home_id']==108:
            angels_pitcher.append(sched[i]['home_probable_pitcher'])
            opposing_pitcher.append(sched[i]['away_probable_pitcher'])
        else:
            opposing_pitcher.append(sched[i]['home_probable_pitcher'])
            angels_pitcher.append(sched[i]['away_probable_pitcher'])
    
        # probable_pitcher.append(sched[i]['home_probable_pitcher'])
    # print(sched[i])
for j in opposing_pitcher:
    pp_ospc = all_pitchers_df.loc[all_pitchers_df['full_name']== j, 'Cluster'].iloc[0]
    # print(pp_ospc)
    opposing_pitcher_cluster.append(pp_ospc)   
    

def prob_pitch():
    # date handling
    date_interval = timedelta(days=3)
    x_current_date = date.today()
    x_end_date = x_current_date + date_interval

    current_date = "%s/%s/%s" % (x_current_date.month, x_current_date.day, x_current_date.year)
    end_date = "%s/%s/%s" % (x_end_date.month, x_end_date.day, x_end_date.year)


    # future_date = todays_date + datetime.timedelta(days=5)
    

    for i in range(0, len(sched)):
        if (sched[i]['home_probable_pitcher']=='') or (sched[i]['away_probable_pitcher']==''):
            break
        else:
            game_date.append(sched[i]['game_date'])
            if sched[i]['home_id']==108:
                angels_pitcher.append(sched[i]['home_probable_pitcher'])
                opposing_pitcher.append(sched[i]['away_probable_pitcher'])
            else:
                opposing_pitcher.append(sched[i]['home_probable_pitcher'])
                angels_pitcher.append(sched[i]['away_probable_pitcher'])
        
            # probable_pitcher.append(sched[i]['home_probable_pitcher'])
        # print(sched[i])
    for j in opposing_pitcher:
        pp_ospc = all_pitchers_df.loc[all_pitchers_df['full_name']== j, 'Cluster'].iloc[0]
        # print(pp_ospc)
        opposing_pitcher_cluster.append(pp_ospc)   
    

    return game_date, opposing_pitcher, opposing_pitcher_cluster



def prob_pitch_df():

    probable_pitchers = prob_pitch()
    prob_pitcher_df = pd.DataFrame(probable_pitchers)

    pp_df = prob_pitcher_df.T
    pp_df.columns = ['date', 'Opponent SP', 'Cluster']

    return pp_df 

p = prob_pitch_df()
# print(p)

sss = statsapi.schedule(start_date = '4/20/2023',team=108)
print(sss)