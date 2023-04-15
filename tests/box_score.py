import statsapi
import pandas as pd
import numpy as np


bx_dict = statsapi.boxscore_data(718614)
bx_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in bx_dict.items()]))
# print(bx_df.columns)
# ['gameId', 'teamInfo, 'playerInfo', 'away', 'home', 'awayBatters',
#        'homeBatters', 'awayBattingTotals', 'homeBattingTotals',
#        'awayBattingNotes', 'homeBattingNotes', 'awayPitchers', 'homePitchers',
#        'awayPitchingTotals', 'homePitchingTotals', 'gameBoxInfo']


bx_gameId = bx_dict['gameId']
df_game = pd.DataFrame([bx_gameId])
"""
                            0
0  2023/04/12/wasmlb-anamlb-1
"""

bx_teamInfo = bx_dict['teamInfo']
df_team = pd.DataFrame([bx_teamInfo][0])

"""
                     away       home
id                   120        108
abbreviation         WSH        LAA
teamName       Nationals     Angels
shortName     Washington  LA Angels
"""

### ['personId'] links to df_playerInfo['id]
bx_homeBatters = bx_dict['homeBatters']
df_homeBatters = pd.DataFrame([bx_homeBatters][0])
"""
        namefield  ab  r  h doubles triples  hr  rbi  sb  bb  k  lob   avg   ops  personId  substitution note            name position   obp   slg battingOrder
0  Angels Batters  AB  R  H      2B      3B  HR  RBI  SB  BB  K  LOB   AVG   OPS         0         False       Angels Batters            OBP   SLG
1      1 Ward  LF   3  0  0       0       0   0    0   0   1  1    5  .292  .831    621493         False                 Ward       LF  .393  .438          100
2     2 Trout  DH   3  0  0       0       0   0    0   0   1  1    1  .262  .960    545361         False                Trout       DH  .436  .524          200
3    3 Rendon  3B   3  0  0       0       0   0    0   0   1  0    2  .176  .536    543685         False               Rendon       3B  .360  .176          300
4   4 Renfroe  RF   3  1  2       1       0   0    0   0   1  0    2  .295  .972    592669         False              Renfroe       RF  .404  .568          400

"""

### Both teams player info ###
### only need ['id', 'fullName']
bx_playerInfo = bx_dict['playerInfo']
df_playerInfo = pd.DataFrame([bx_playerInfo][0])
df_playerInfo = df_playerInfo.T
"""
             id         fullName boxscoreName
ID621433  621433   Brett Phillips  Phillips, B
ID656180  656180      Riley Adams     Adams, R
ID592866  592866  Trevor Williams     Williams
ID642136  642136      Matt Thaiss       Thaiss
ID680686  680686      Josiah Gray     Gray, Js

"""


bx_homePitchers = bx_dict['homePitchers']
df_homePitchers = pd.DataFrame([bx_homePitchers][0])
df_homePitchers.rename(columns=df_homePitchers.iloc[0], inplace=True)
df_homePitchers.drop([0], inplace=True)
df_homePitchers.rename(columns={0:'id', '':'note'}, inplace=True)
print(df_homePitchers)
"""
         namefield   ip  h  r  er  bb  k  hr   era   p   s                name  personId      note
0   Angels Pitchers   IP  H  R  ER  BB  K  HR   ERA   P   S  Nationals Pitchers         0
1  Ohtani  (W, 2-0)  7.0  1  0   0   5  6   0  0.47  92  55              Ohtani    660271  (W, 2-0)
2   Quijada  (H, 3)  1.0  0  0   0   0  0   0  0.00  15  11             Quijada    650671    (H, 3)
3   Estévez  (S, 1)  1.0  0  0   0   1  0   0  1.80  17  11             Estévez    608032    (S, 1)

"""
# print(df.columns)

# print(type(bx_df))
# print(bx_df.columns)
# print(bx_df['playerInfo'][0])

# print(df.columns)
# df.rename(columns=df.iloc[0], inplace = True)
# df.drop([0], inplace = True)
# print(df.head())
# for i,j in enumerate(df.columns):
#     print(i, j)