import random

class RandomAgent:
    def __init__(self,env):

        self.env = env

    def choose_move(self,moves):

        return random.choice(moves)
