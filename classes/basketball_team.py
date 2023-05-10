import random

from classes.player import Player

class BasketballTeam:

    name: str

    list_of_players: list[Player]
    floor_players = list[Player]

    _2PT_field_goal_attempts = 0
    _3PT_field_goal_attempts = 0
    _2PT_field_goals_made = 0
    _3PT_field_goals_made = 0
    _2PT_field_goal_percentage = 0
    _3PT_field_goal_percentage = 0
    total_PTS = 0

    def __init__(self, name, list_of_players, floor_players):
        self.name = name
        self.list_of_players = list_of_players
        self.floor_players = floor_players

    def choose_player_for_attack(self):
        random_player_index = random.randint(0, 4)
        player = self.floor_players[random_player_index]
        return player
    
    def get_total_points(self):
        points_by_player = [player.total_PTS for player in self.list_of_players]
        total_points = sum(points_by_player)
        return total_points