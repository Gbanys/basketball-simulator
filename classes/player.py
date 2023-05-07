import numpy as np
import pandas as pd
import random
import warnings
import time

warnings.filterwarnings("ignore")


def min_max_normalize(value, min, max):
    normalized_value = (value - min) / (max - min)
    return normalized_value


class Player:

    name: str

    dunk_ability: float
    layup_ability: float
    free_throw_ability: float
    mid_range_ability: float
    three_pointer_ability: float
    energy: float
    energy_depletion_rate: float

    dunk_tendency: float
    layup_tendency: float
    mid_range_tendency: float
    three_pointer_tendency: float

    _2PT_field_goal_attempts = 0
    _3PT_field_goal_attempts = 0
    _2PT_field_goals_made = 0
    _3PT_field_goals_made = 0
    _2PT_field_goal_percentage = 0
    _3PT_field_goal_percentage = 0
    total_PTS = 0

    def __init__(
            self, 
            name,
            dunk_ability, 
            layup_ability, 
            free_throw_ability, 
            mid_range_ability, 
            three_pointer_ability, 
            energy,
            energy_depletion_rate,
            dunk_tendency,
            layup_tendency,
            mid_range_tendency,
            three_pointer_tendency
        ):

        self.name = name
        self.dunk_ability = min_max_normalize(dunk_ability, 0, 100)
        self.layup_ability = min_max_normalize(layup_ability, 0, 100)
        self.free_throw_ability = min_max_normalize(free_throw_ability, 0, 100)
        self.mid_range_ability = min_max_normalize(mid_range_ability, 0, 100)
        self.three_pointer_ability = min_max_normalize(three_pointer_ability, 0, 100)
        self.energy = min_max_normalize(energy, 0, 100)
        self.energy_depletion_rate = energy_depletion_rate
        self.dunk_tendency = dunk_tendency
        self.layup_tendency = layup_tendency
        self.mid_range_tendency = mid_range_tendency
        self.three_pointer_tendency = three_pointer_tendency

        return


    def decide_on_action(self) -> str:

        total = self.dunk_tendency + self.layup_tendency + self.mid_range_tendency + self.three_pointer_tendency

        p_dunk = self.dunk_tendency / total
        p_layup = self.layup_tendency / total
        p_mid_range = self.mid_range_tendency / total
        p_three_pointer = self.three_pointer_tendency / total

        choice = np.random.choice(['dunk', 'layup', 'mid_range', 'three_pointer'], p = [p_dunk, p_layup, p_mid_range, p_three_pointer])

        return choice

    def dunk(self, dunk_ml_model):
        print(f'{self.name} attempts a dunk.')
        time.sleep(2)
        outcome = dunk_ml_model.predict(np.array([[self.dunk_ability, self.energy]]))
        if outcome == 1:
            print(f'{self.name} dunks it successfully.')
        else:
            print(f'{self.name} misses the dunk!.\n')
        time.sleep(2)
        return outcome
    
    
    def do_layup(self, layup_ml_model):
        print(f'{self.name} attempts a layup.')
        time.sleep(2)
        outcome = layup_ml_model.predict(np.array([[self.layup_ability, self.energy]]))
        if outcome == 1:
            print(f'{self.name} makes a simple layup.')
        else:
            print(f'{self.name} misses the layup!.\n')
        time.sleep(2)
        return outcome
    
    
    def shoot_free_throw(self, free_throw_ml_model):
        shot_quality = min_max_normalize(random.randint(10, 90), 0, 100)
        outcome = free_throw_ml_model.predict(np.array([[self.free_throw_ability, self.energy, shot_quality]]))
        if outcome == 1:
            print(f'{self.name} makes the free throw.')
        else:
            print(f'{self.name} misses the free throw.\n')
        return outcome
    
    
    def shoot_mid_range(self, mid_range_ml_model: pd.DataFrame):
        print(f'{self.name} shoots from mid range.')
        time.sleep(2)
        shot_quality = min_max_normalize(random.randint(10, 90), 0, 100)
        distance = min_max_normalize(random.randint(15, 23), 15, 23)
        outcome = mid_range_ml_model.predict(np.array([[distance, self.mid_range_ability, self.energy, shot_quality]]))
        if outcome == 1:
            print(f'{self.name} makes the mid-range jumper.')
        else:
            print(f'{self.name} misses the mid-range jumper.\n')
        time.sleep(2)
        return outcome
    
    
    def shoot_three_pointer(self, three_pointer_ml_model: pd.DataFrame):
        print(f'{self.name} attempts a three point shot.')
        time.sleep(2)
        shot_quality = min_max_normalize(random.randint(10, 90), 0, 100)
        distance = min_max_normalize(random.randint(23, 28), 23, 28)
        outcome = three_pointer_ml_model.predict(np.array([[distance, self.three_pointer_ability, self.energy, shot_quality]]))
        if outcome == 1:
            print(f'{self.name} makes the three pointer!.')
        else:
            print(f'{self.name} misses the three.\n')
        time.sleep(2)
        return outcome
    



