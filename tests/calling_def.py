import statsapi
import pandas as pd
import numpy as np
# from __init__ import boxscore_data, schedule 

# sched = statsapi.schedule(start_date='04/13/2023',end_date='04/29/2023',team=108)
"""
import pandas as pd
import numpy as np

d = dict( A = np.array([1,2]), B = np.array([1,2,3,4]) )
    
pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
"""


# s = statsapi.schedule(start_date = '4/1/2023', end_date = '5/1/2023', team=108)
# print(s)

# l = statsapi.linescore(718614)
# print(l)
"""
Final     1 2 3 4 5 6 7 8 9  R   H   E  
Nationals 0 0 0 0 0 0 0 0 0  0   1   0
Angels    0 0 0 1 0 1 0 0 0  2   5   0
"""

p = statsapi.player_stat_data(545361)  # dict
# p_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in p.items() ]))

# print(p_df.columns)
player_stats = p['stats']
player_stats_df = pd.DataFrame([player_stats][0])
# print(player_stats_df.iloc[2])

s = statsapi.schedule(start_date = '4/17/2023', team=108)
print(len(s))
# print(s['game_id'][0])

# print(type(p))
empt = {}


for d in s:    
    for key in d:
        print(f"{key}:{d[key]}")
        empt[key] = d[key]

# print(empt['game_id'])
"""
my_dict = {"username": "XYZ", "email": "xyz@gmail.com", "location":"Mumbai"}

my_dict['name']='Nick'

print(my_dict)
"""

bx_d = statsapi.boxscore_data(718540)

angels = bx_d['awayBatters']

angels_df = pd.DataFrame([angels][0])

print(angels_df)



