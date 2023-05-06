import pandas as pd
import numpy as np
import random
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_dunk_model():

    scaled_player_ability, shot_made_by_ability = generate_random_player_ability_data()
    scaled_player_energy, shot_made_by_energy = generate_random_player_energy_data()

    dunk_dataframe = pd.DataFrame({'scaled_player_ability' : [scaled_player_ability[i][0] for i in range(0, len(scaled_player_ability))],
            'scaled_player_energy' : [scaled_player_energy[i][0] for i in range(0, len(scaled_player_energy))],
            'shot_made_by_ability' : shot_made_by_ability,
            'shot_made_by_energy' : shot_made_by_energy,
    })
    def determine_if_shot_made(value):
        if value == 1:
            return 1
        else:
            return 0
        
    dunk_dataframe['shot_made_overall'] = dunk_dataframe[['shot_made_by_ability', 'shot_made_by_energy']]\
    .apply(lambda x: np.mean(x), axis=1)
    dunk_dataframe['shot_made_overall'] = dunk_dataframe['shot_made_overall'].apply(determine_if_shot_made)

    input_features = dunk_dataframe[['scaled_player_ability', 'scaled_player_energy']]
    shot_made_overall = dunk_dataframe['shot_made_overall']

    X_train, X_test, y_train, y_test = train_test_split(input_features, shot_made_overall)

    dunk_model = LogisticRegression().fit(X_train, y_train)
    score = dunk_model.score(X_test, y_test)

    print("Model training SUCCESSFUL:")
    print(f'Model score: {score}')

    pickle.dump(dunk_model, open('models/dunk_model.pkl', 'wb'))


def generate_random_player_ability_data():

    scaler = MinMaxScaler()

    player_ability = np.array([random.uniform(0, 100) for i in range(0, 2000)])
    scaled_player_ability = scaler.fit_transform(player_ability.reshape(-1,1))

    shot_made_by_ability = []

    for i in range(0, len(scaled_player_ability)):
        x_value = scaled_player_ability[i][0]
        p_shot_made = x_value**0.4
        p_shot_not_made = 1 - p_shot_made
        y_value = np.random.choice([1, 0], p=[p_shot_made, p_shot_not_made])
        shot_made_by_ability.append(y_value)

    return scaled_player_ability, shot_made_by_ability


def generate_random_player_energy_data():

    scaler = MinMaxScaler()

    player_energy = np.array([random.uniform(0, 100) for i in range(0, 2000)])
    scaled_player_energy = scaler.fit_transform(player_energy.reshape(-1,1))

    shot_made_by_energy = []

    for i in range(0, len(scaled_player_energy)):
        x_value = scaled_player_energy[i][0]
        p_shot_made = np.exp(x_value / 5) - 0.8
        p_shot_not_made = 1 - p_shot_made
        y_value = np.random.choice([1, 0], p=[p_shot_made, p_shot_not_made])
        shot_made_by_energy.append(y_value)

    return scaled_player_energy, shot_made_by_energy


train_dunk_model()