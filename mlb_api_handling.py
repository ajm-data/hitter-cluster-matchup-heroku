import statsapi
import pandas as pd
import datetime
from datetime import date, datetime, timedelta

# sched = statsapi.schedule(start_date='04/13/2023',end_date='04/29/2023',team=108)

all_pitchers = pd.read_csv('Named_Clustered_Metric.csv')
all_pitchers_df = pd.DataFrame(all_pitchers)
all_pitchers_df['full_name'] = all_pitchers_df['first_name'] + ' ' + all_pitchers_df['last_name']
all_pitchers_df['full_name'] = all_pitchers_df['full_name'].str.strip()

all_pitchers_df = all_pitchers_df[['Cluster', 'full_name']]

    # date handling
date_interval = timedelta(days=5)
x_current_date = date.today()
x_end_date = x_current_date + date_interval

current_date = "%s/%s/%s" % (x_current_date.month, x_current_date.day, x_current_date.year)
end_date = "%s/%s/%s" % (x_end_date.month, x_end_date.day, x_end_date.year)

def prob_pitch():
    # date handling
    date_interval = timedelta(days=5)
    x_current_date = date.today()
    x_end_date = x_current_date + date_interval

    current_date = "%s/%s/%s" % (x_current_date.month, x_current_date.day, x_current_date.year)
    end_date = "%s/%s/%s" % (x_end_date.month, x_end_date.day, x_end_date.year)


    # future_date = todays_date + datetime.timedelta(days=5)
    
    sched = statsapi.schedule(start_date=current_date,end_date=end_date,team=108)
    probable_pitcher = []
    angels_pitcher = []
    opposing_pitcher = []
    game_date = []
    opposing_pitcher_cluster = []
    opposing_team = []
    gameId = {}

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
    # for j in opposing_pitcher:
    #     pp_ospc = all_pitchers_df.loc[all_pitchers_df['full_name']== j, 'Cluster']
    #     opposing_pitcher_cluster.append(pp_ospc)   

    return game_date, opposing_pitcher, opposing_pitcher_cluster

def prob_pitch_df():

    probable_pitchers = prob_pitch()

    # for i in probable_pitchers:
    #     if isinstance(i, str):
    #         probable_pitchers
    prob_pitcher_df = pd.DataFrame(probable_pitchers)

    pp_df = prob_pitcher_df.T
    pp_df.columns = ['date', 'full_name', 'Cluster']

    
    plm = pd.merge(pp_df, all_pitchers_df, on='full_name')

    plm = plm.rename(columns={'Cluster_y':'Cluster'})
    plm = plm.drop(columns='Cluster_x')
    
    # pp_ospc = all_pitchers_df.loc[all_pitchers_df['full_name']== pp_df['Opponent SP'], 'Cluster']
    # print(type(pp_ospc))
    return plm


def live_box():
    gameId = {}
    tod = statsapi.schedule(start_date=current_date,team=108)
    for d in tod:
        for key in d:
            # print(f"{key}:{d[key]}")
            gameId[key] = d[key]
    
    game_id = gameId['game_id']

    bx_d = statsapi.boxscore_data(game_id)

    angels = bx_d['awayBatters']

    angels_df = pd.DataFrame([angels][0])

    adf_col = ['namefield', 'ab', 'h', 'r', 'hr', 'rbi', 'obp', 'slg','personId']
    angels_df = angels_df[adf_col]

    return angels_df


xlv = live_box()
print(xlv)
# x = prob_pitch_df()
# print(x)




# print(prob_pitch())


# def prob_pitch_cluster():

#     xxxx = prob_pitch_df()

#     xxxx.columns = [xxxx.columns, 'Opponent SP Cluster']
#     for i in xxxx['Opponent SP']:
#         pp_ospc = all_pitchers_df.loc[all_pitchers_df['full_name']== i, 'Cluster'].iloc[0]
#         ppl.append(pp_ospc)
    
#     ppl_df_raw = pd.DataFrame(ppl)
#     ppl_df = ppl_df_raw.T

#     xxxx['Opponent SP Cluster'] = ppl_df

#     return xxxx

# yoyo = prob_pitch_cluster()

# print(yoyo)








            


# def create_prob_pitch_df():
#     for i in range(0, 8):  
#         prob_pitcher_df.append({'Date': sched[i]['game_date']}, ignore_index=True)
        # if sched[i]['away_id']==108:
        #     prob_pitcher_df.append({'Angels Pitcher': sched[i]['away_probable_pitcher']}, ignore_index=True)
        #     prob_pitcher_df.append({'Opposing Pitcher': sched[i]['home_probable_pitcher']}, ignore_index=True)
        # elif sched[i]['home_id']==108:
        #     prob_pitcher_df.append({'Angels Pitcher': sched[i]['home_probable_pitcher']}, ignore_index=True)
        #     prob_pitcher_df.append({'Opposing Pitcher': sched[i]['away_probable_pitcher']}, ignore_index=True)
        # else:
        #     print('wtf is this')




# xdf = create_prob_pitch_df()

# print(xdf)
           






# def angels_probable():
#     for i in range(0, len(sched)):
#         if sched[i]['away_id']== 108:
#             print('the angels are the away team')
#         elif sched[i]['home_id']==108:
#             print('the angels are the home team')
#         else:
#             break

# angels_probable()