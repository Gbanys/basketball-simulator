from datetime import datetime
from datetime import timedelta
from classes.player import Player
import pickle

time_start = datetime.now()
time_end = datetime.now() + timedelta(0, 60)

dunk_ml_model = pickle.load(open('models/dunk_model.pkl', 'rb'))
layup_ml_model = pickle.load(open('models/layup_model.pkl', 'rb'))
free_throw_ml_model = pickle.load(open('models/free_throw_model.pkl', 'rb'))
mid_range_ml_model = pickle.load(open('models/mid_range_model.pkl', 'rb'))
three_pointer_ml_model = pickle.load(open('models/three_pointer_model.pkl', 'rb'))

player = Player(
    name = "Abril Diaz",
    dunk_ability=60,
    layup_ability=60,
    free_throw_ability=80,
    mid_range_ability=83,
    three_pointer_ability=80,
    energy = 90,
    energy_depletion_rate=2,
    defending=60,
    dunk_tendency=20,
    layup_tendency=80,
    mid_range_tendency=70,
    three_pointer_tendency=30,
    opponent_on_ball_defending = 80,
    position = 'PG',
    team_name='Eagles'
)

current_time = datetime.now()

while current_time < time_end:

    current_time = datetime.now()

    choice = player.decide_on_action()

    if choice == 'dunk':
        outcome = player.dunk(dunk_ml_model)
        player.total_PTS += 2 if outcome == 1 else 0
        player._2PT_field_goals_made += 1 if outcome == 1 else 0
        player._2PT_field_goal_attempts += 1
    elif choice == 'layup':
        outcome = player.do_layup(layup_ml_model)
        player.total_PTS += 2 if outcome == 1 else 0
        player._2PT_field_goals_made += 1 if outcome == 1 else 0
        player._2PT_field_goal_attempts += 1
    elif choice == 'mid_range':
        outcome = player.shoot_mid_range(mid_range_ml_model)
        player.total_PTS += 2 if outcome == 1 else 0
        player._2PT_field_goals_made += 1 if outcome == 1 else 0
        player._2PT_field_goal_attempts += 1
    elif choice == 'three_pointer':
        outcome = player.shoot_three_pointer(three_pointer_ml_model)
        player.total_PTS += 3 if outcome == 1 else 0
        player._3PT_field_goals_made += 1 if outcome == 1 else 0
        player._3PT_field_goal_attempts += 1

    if player.energy > 20:
        player.energy = player.energy - player.energy_depletion_rate


statistics = {
    '2PT_field_goals_attempted' : player._2PT_field_goal_attempts,
    '2PT_field_goals_made' : player._2PT_field_goals_made,
    '2PT_field_goal_percentage' : (player._2PT_field_goals_made / player._2PT_field_goal_attempts) * 100,
    '3PT_field_goals_attempted' : player._3PT_field_goal_attempts,
    '3PT_field_goals_made' : player._3PT_field_goals_made,
    '3PT_field_goal_percentage' : (player._3PT_field_goals_made / player._3PT_field_goal_attempts) * 100,
    'Total Points' : player.total_PTS
}
print(statistics)