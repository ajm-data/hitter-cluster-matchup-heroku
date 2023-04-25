import pandas as pd


def player_map_for_table():
    player_id_map_csv = pd.read_csv('SFBB Player ID Map - PLAYERIDMAP.csv')
    player_id_map_df = pd.DataFrame(player_id_map_csv)
    player_id_map_df.columns = [x.lower() for x in player_id_map_df.columns]
    

    p_id_m_cols = [col.lower() for col in ['IDPLAYER', 'MLBID', 'PLAYERNAME', 'FIRSTNAME', 'LASTNAME', 'TEAM',
                'POS', 'MLBNAME', 'BATS', 'THROWS', 'ALLPOS',
                'ROTOWIREID', 'BREFID', 'FANDUELID',
                'ROTOWIRENAME','FANDUELNAME','DRAFTKINGSNAME']]

    active_player_map = player_id_map_df[p_id_m_cols].loc[ (player_id_map_df['active'] == 'Y')]
    
    return active_player_map


