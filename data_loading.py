import pandas as pd

from classes.player import Player

def load_player_data(team_name: str) -> list[Player]:

    player_dataframe = pd.read_csv("player_data.csv")
    player_dataframe = player_dataframe[player_dataframe['team_name'] == team_name]

    players_list = []
    for index in player_dataframe.index.values:
        player = Player(
            player_dataframe.loc[index, 'name'],
            player_dataframe.loc[index, 'dunk_ability'],
            player_dataframe.loc[index, 'layup_ability'],
            player_dataframe.loc[index, 'free_throw_ability'],
            player_dataframe.loc[index, 'mid_range_ability'],
            player_dataframe.loc[index, 'three_pointer_ability'],
            player_dataframe.loc[index, 'energy'],
            player_dataframe.loc[index, 'energy_depletion_rate'],
            player_dataframe.loc[index, 'defending'],
            player_dataframe.loc[index, 'dunk_tendency'],
            player_dataframe.loc[index, 'layup_tendency'],
            player_dataframe.loc[index, 'mid_range_tendency'],
            player_dataframe.loc[index, 'three_pointer_tendency'],
            player_dataframe.loc[index, 'opponent_on_ball_defending'],
            player_dataframe.loc[index, 'position'],
            player_dataframe.loc[index, 'team_name']
        )
        players_list.append(player)

    return players_list

