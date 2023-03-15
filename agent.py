import torch
import torch.nn as nn
import numpy as np

from constants import *
import board
import random

from replay_buffer import ReplayBuffer

PIECE_VALUE = [1,3,3,5,9,100]

class Agent:
    def __init__(self,env,color,config):
        
        self.env = env
        self.color = color
        self.config = config

        n_layers = self.config.n_layers
        layer_size = self.config.layer_size

        input_size = BOARD_SIZE * BOARD_SIZE + 1

        network = nn.Sequential()
        prev_size = input_size
        for i in range(n_layers):
            network.append(nn.Linear(prev_size,size))
            network.append(nn.ReLU())
            prev_size = size
        network.append(nn.Linear(prev_size,1))
        self.q_network = network

        network = nn.Sequential()
        prev_size = input_size
        for i in range(n_layers):
            network.append(nn.Linear(prev_size,size))
            network.append(nn.ReLU())
            prev_size = size
        network.append(nn.Linear(prev_size,1))
        self.target_network = network

        self.optimizer = torch.optim.Adam(self.q_network.paramters(), 
                lr=self.config.learning_rate)
        self.memory = ReplayMemory(self.config.batch_size)


    def evaluate_moves(self,moves):
        evals = np.zeros(np.size(moves))
        for i in range(np.size(moves)):
            B_next = board.Board(self.env)
            B_next.make_move(moves[i])
            turn = (self.color+1)%2
            evals[i] = self.evaluate(B_next,turn)
        return evals.detach().numpy()

    def evaluate(self,B,turn):
        state = board.flatten(B)
        state.append(turn)
        val = self.q_network(np2torch(state))
        return val

    def calculate_rewards(self,moves):
        rewards = np.zeros(np.size(moves))
        for i in range(np.size(moves)):
            rewards[i] = self.calculate_reward(move)
        return rewards

    def calculate_reward(self,move):
        out = 0
        B_next = board.Board(self.env)
        B_next.makeMove(move)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                factor = 1.0
                if B_next.color(i,j) == BLACK:
                    factor = -1.0
                val = PIECE_VALUE[(B_next.squares[i][j]-1)//2]
                out += val * factor
                factor = 1.0
                if self.env.color(i,j) == BLACK:
                    factor = -1.0
                val = PIECE_VALUE[(self.env.squares[i][j]-1)//2]
                out -= val * factor
        return out

    def choose_move(self,moves,eps):

        if random.uniform(0,1) < eps:
            return random.choice(moves)

        vals = self.evaluate_moves(moves)
        if self.color == BLACK:
            vals = -1.0*vals # Black wants to minimize
        best_move = moves[np.argmax(vals)]
        return best_move

    def save(self):
        torch.save(self.q_network.state_dict(), self.config.model_output)



# from A3 starter code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x

