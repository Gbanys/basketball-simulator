from datetime import datetime
from datetime import timedelta
from classes.basketball_team import BasketballTeam
import pickle
from data_loading import load_player_data

time_start = datetime.now()
time_end = datetime.now() + timedelta(0, 400)

dunk_ml_model = pickle.load(open('models/dunk_model.pkl', 'rb'))
layup_ml_model = pickle.load(open('models/layup_model.pkl', 'rb'))
free_throw_ml_model = pickle.load(open('models/free_throw_model.pkl', 'rb'))
mid_range_ml_model = pickle.load(open('models/mid_range_model.pkl', 'rb'))
three_pointer_ml_model = pickle.load(open('models/three_pointer_model.pkl', 'rb'))

home_players = load_player_data("Eagles")
home_team = BasketballTeam("Eagles", home_players, home_players)
away_players = load_player_data("Barcelona")
away_team = BasketballTeam("Barcelona", away_players, away_players)

def assign_opponents(team_one, team_two):
    for team_one_player in team_one.list_of_players:
        for team_two_player in team_two.list_of_players:
            if team_one_player.position == team_two_player.position:
                index = team_one.list_of_players.index(team_one_player)
                team_one.list_of_players[index].opponent_on_ball_defending = team_two_player.defending
    return team_one

def add_2pt_to_player_in_team(team, player, outcome):
    list_index = team.list_of_players.index(player)
    team.list_of_players[list_index].total_PTS += 2 if outcome == 1 else 0
    team.list_of_players[list_index]._2PT_field_goals_made += 1 if outcome == 1 else 0
    team.list_of_players[list_index]._2PT_field_goal_attempts += 1
    return team

def add_3pt_to_player_in_team(team, player, outcome):
    list_index = team.list_of_players.index(player)
    team.list_of_players[list_index].total_PTS += 3 if outcome == 1 else 0
    team.list_of_players[list_index]._3PT_field_goals_made += 1 if outcome == 1 else 0
    team.list_of_players[list_index]._3PT_field_goal_attempts += 1
    return team

home_team = assign_opponents(home_team, away_team)
away_team = assign_opponents(away_team, home_team)

current_time = datetime.now()

current_team_turn = True

while current_time < time_end:

    if current_team_turn:
        print("Home Team")
        player = home_team.choose_player_for_attack()
    else:
        print("Away Team")
        player = away_team.choose_player_for_attack()

    current_time = datetime.now()

    choice = player.decide_on_action()

    if choice == 'dunk':
        outcome = player.dunk(dunk_ml_model)
        home_team = add_2pt_to_player_in_team(home_team, player, outcome) if current_team_turn else home_team
        away_team = add_2pt_to_player_in_team(away_team, player, outcome) if not current_team_turn else away_team
    elif choice == 'layup':
        outcome = player.do_layup(layup_ml_model)
        home_team = add_2pt_to_player_in_team(home_team, player, outcome) if current_team_turn else home_team
        away_team = add_2pt_to_player_in_team(away_team, player, outcome) if not current_team_turn else away_team
    elif choice == 'mid_range':
        outcome = player.shoot_mid_range(mid_range_ml_model)
        home_team = add_2pt_to_player_in_team(home_team, player, outcome) if current_team_turn else home_team
        away_team = add_2pt_to_player_in_team(away_team, player, outcome) if not current_team_turn else away_team
    elif choice == 'three_pointer':
        outcome = player.shoot_three_pointer(three_pointer_ml_model)
        home_team = add_3pt_to_player_in_team(home_team, player, outcome) if current_team_turn else home_team
        away_team = add_3pt_to_player_in_team(away_team, player, outcome) if not current_team_turn else away_team

    if player.energy > 20:
        player.energy = player.energy - player.energy_depletion_rate

    current_team_turn = not current_team_turn

    print(f'{home_team.name} {home_team.get_total_points()}:{away_team.get_total_points()} {away_team.name}')