import pandas as pd
import numpy as np
import random
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_three_pointer_model():

    scaled_distance_from_basket, shot_made_by_distance = generate_random_distance_data()
    scaled_player_ability, shot_made_by_ability = generate_random_player_ability_data()
    scaled_player_energy, shot_made_by_energy = generate_random_player_energy_data()
    scaled_shot_quality, shot_made_by_shot_quality = generate_random_shot_quality_data()

    three_pointer_dataframe = pd.DataFrame({'scaled_distance_from_basket' : [scaled_distance_from_basket[i][0] for i in range(0, len(scaled_distance_from_basket))],
            'scaled_player_ability' : [scaled_player_ability[i][0] for i in range(0, len(scaled_player_ability))],
            'scaled_player_energy' : [scaled_player_energy[i][0] for i in range(0, len(scaled_player_energy))],
            'scaled_shot_quality' : [scaled_shot_quality[i][0] for i in range(0, len(scaled_shot_quality))],
            'shot_made_by_distance' : shot_made_by_distance,
            'shot_made_by_ability' : shot_made_by_ability,
            'shot_made_by_energy' : shot_made_by_energy,
            'shot_made_by_shot_quality' : shot_made_by_shot_quality
    })
    def determine_if_shot_made(value):
        if value >= 0.5:
            return 1
        else:
            return 0

    three_pointer_dataframe['shot_made_overall'] = three_pointer_dataframe[['shot_made_by_distance', 'shot_made_by_ability', 'shot_made_by_energy', 'shot_made_by_shot_quality']]\
    .apply(lambda x: np.mean(x), axis=1)
    three_pointer_dataframe['shot_made_overall'] = three_pointer_dataframe['shot_made_overall'].apply(determine_if_shot_made)

    input_features = three_pointer_dataframe[['scaled_distance_from_basket', 'scaled_player_ability', 'scaled_player_energy', 'scaled_shot_quality']]
    shot_made_overall = three_pointer_dataframe['shot_made_overall']

    X_train, X_test, y_train, y_test = train_test_split(input_features, shot_made_overall)

    three_pointer_model = LogisticRegression().fit(X_train, y_train)
    score = three_pointer_model.score(X_test, y_test)

    print("Model training SUCCESSFUL:")
    print(f'Model score: {score}')

    pickle.dump(three_pointer_model, open('models/three_pointer_model.pkl', 'wb'))



def generate_random_distance_data():

    scaler = MinMaxScaler()
    distance_from_basket = np.array([random.uniform(15, 23) for i in range(0, 2000)])
    scaled_distance_from_basket = scaler.fit_transform(distance_from_basket.reshape(-1,1))

    shot_made_by_distance = []

    for i in range(0, len(scaled_distance_from_basket)):
        x_value = scaled_distance_from_basket[i][0]
        p_shot_made = np.exp(-x_value / 3) - 0.3
        p_shot_not_made = 1 - p_shot_made
        y_value = np.random.choice([1, 0], p=[p_shot_made, p_shot_not_made])
        shot_made_by_distance.append(y_value)

    return scaled_distance_from_basket, shot_made_by_distance


def generate_random_player_ability_data():

    scaler = MinMaxScaler()

    player_ability = np.array([random.uniform(0, 100) for i in range(0, 2000)])
    scaled_player_ability = scaler.fit_transform(player_ability.reshape(-1,1))

    shot_made_by_ability = []

    for i in range(0, len(scaled_player_ability)):
        x_value = scaled_player_ability[i][0]
        p_shot_made = (np.exp(5 * x_value) - 0) / (155 - 0)
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


def generate_random_shot_quality_data():

    scaler = MinMaxScaler()

    shot_quality = np.array([random.uniform(0, 100) for i in range(0, 2000)])
    scaled_shot_quality = scaler.fit_transform(shot_quality.reshape(-1,1))

    shot_made_by_shot_quality = []

    for i in range(0, len(scaled_shot_quality)):
        x_value = scaled_shot_quality[i][0]
        p_shot_made = (np.exp(5 * x_value) - 0) / (155 - 0)
        p_shot_not_made = 1 - p_shot_made
        y_value = np.random.choice([1, 0], p=[p_shot_made, p_shot_not_made])
        shot_made_by_shot_quality.append(y_value)

    return scaled_shot_quality, shot_made_by_shot_quality


train_three_pointer_model()