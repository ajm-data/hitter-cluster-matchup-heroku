import statsapi
import pandas as pd


# sched = statsapi.schedule(start_date='04/13/2023',end_date='04/29/2023',team=108)

all_pitchers = pd.read_csv('Named_Clustered_Metric.csv')
all_pitchers_df = pd.DataFrame(all_pitchers)
all_pitchers_df['full_name'] = all_pitchers_df['first_name'] + ' ' + all_pitchers_df['last_name']
all_pitchers_df['full_name'] = all_pitchers_df['full_name'].str.strip()

all_pitchers_df = all_pitchers_df[['Cluster', 'full_name']]


def prob_pitch():
    sched = statsapi.schedule(start_date='04/13/2023',end_date='04/29/2023',team=108)
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

    for j in opposing_pitcher:
        pp_ospc = all_pitchers_df.loc[all_pitchers_df['full_name']== j, 'Cluster'].iloc[0]
        opposing_pitcher_cluster.append(pp_ospc)   

    return game_date, opposing_pitcher, opposing_pitcher_cluster

def prob_pitch_df():

    probable_pitchers = prob_pitch()
    prob_pitcher_df = pd.DataFrame(probable_pitchers)

    pp_df = prob_pitcher_df.T
    pp_df.columns = ['date', 'Opponent SP', 'Cluster']

    return pp_df 




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