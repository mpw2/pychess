import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
from pathlib import Path
import pickle
import random

from constants import *
import board
import config

from replay_buffer import ReplayBuffer
from schedule import LinearExploration, LinearSchedule

# values = [empty square, pawn, knight, bishop, rook, queen, king]
PIECE_VALUE = [0,1,3,3,5,9,100]

class Agent:
    def __init__(self,env,config):
        
        self.env = env
        self.config = config

        n_layers = self.config.n_layers
        layers_size = self.config.layers_size

        input_size = BOARD_SIZE * BOARD_SIZE * NUM_PIECES + 1

        network = nn.Sequential()
        prev_size = input_size
        for i in range(n_layers):
            size = layers_size[i]
            network.append(nn.Linear(prev_size,size))
            network.append(nn.ReLU())
            prev_size = size
        network.append(nn.Linear(prev_size,1))
        self.q_network = network

        network = nn.Sequential()
        prev_size = input_size
        for i in range(n_layers):
            size = layers_size[i]
            network.append(nn.Linear(prev_size,size))
            network.append(nn.ReLU())
            prev_size = size
        network.append(nn.Linear(prev_size,1))
        self.target_network = network

        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        self.memory = ReplayBuffer(self.config.buffer_size)
        #self.capture_memory = ReplayBuffer(self.config.buffer_size)

        # exploration strategy
        self.exp_schedule = LinearExploration(
           self.env, self.config.eps_begin, self.config.eps_end, self.config.eps_nsteps         
        )

        # output directory
        if not os.path.exists(self.config.output_path):
            os.makedirs(self.config.output_path)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")

        # load model weights if config option specified
        if hasattr(self.config, "load_path"):
            print("Loading parameters from file:", self.config.load_path)
            load_path = Path(self.config.load_path)
            self.q_network.load_state_dict(torch.load(load_path, map_location="cpu"))
            print("Load successful!")


    def evaluate_moves(self,B,moves,network):
        evals = torch.zeros(len(moves))
        for i in range(np.size(moves)):
            B_next = board.Board(B)
            B_next.makeMove(moves[i])
            turn = (B.toMove+1)%2
            evals[i] = self.evaluate(B_next,turn,network)
        material_gains = self.calculate_rewards(moves,B)
        evals += np2torch(material_gains)
        return evals

    def evaluate(self,B,turn,network):
        state = B.flatten()
        state = np.append(state,turn)
        if network == "q_network":
            val = self.q_network(np2torch(state))
        elif network == "target_network":
            val = self.target_network(np2torch(state))
        return val

    def calculate_rewards(self,moves,B=None):
        if B is None:
            B = self.env
        rewards = np.zeros(np.size(moves))
        for i in range(np.size(moves)):
            rewards[i] = self.calculate_reward(moves[i],B=B)
        return rewards

    def calculate_reward(self,move,B=None):
        """
        computes the material advantage before and after making the move
        """
        out = 0
        if B is None:
            B = self.env
        B_next = board.Board(B)
        B_next.makeMove(move)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                factor = 1.0
                if B_next.color(i,j) == BLACK:
                    factor = -1.0
                val = PIECE_VALUE[(B_next.squares[i][j]+1)//2]
                out += val * factor
                factor = 1.0
                if self.env.color(i,j) == BLACK:
                    factor = -1.0
                val = PIECE_VALUE[(self.env.squares[i][j]+1)//2]
                out -= val * factor
        return out

    def choose_move(self,moves):
        # factor = 1.0
        # if self.env.toMove == BLACK:
        #     factor = -1.0

        # rewards = self.calculate_rewards(moves)
        # rewards *= factor
        # vals = np.zeros(len(moves))
        # for i,m in enumerate(moves):
        #     B_next = board.Board(self.env)
        #     B_next.makeMove(m)
        #     next_moves = B_next.allLegalMoves(B_next.toMove)
        #     next_vals = self.evaluate_moves(B_next,next_moves,"q_network")
        #     next_vals *= factor
        #     vals[i] = rewards[i] + np.min(next_vals)

        vals = self.evaluate_moves(self.env,moves,"q_network")
        if self.env.toMove == BLACK:
            vals = -1.0*vals # Black wants to minimize
        best_move = moves[torch.argmax(vals)]
        return best_move

    def get_q_values(self, state, color, network):
        """
        Args:
            state: (list)
                list of boards with len(state) = batch_size
            color: (list)
                list of colors that we are evaluating for
            network: (str)
                name of the network we want to use for the forward pass,
                either "q_network" or "target_network"
        
        Returns:
            out: (list)
                list of torch tensors with len(out) = batch_size
        """

        out = []
        
        for s,c in zip(state,color):
            moves = s.allLegalMoves(s.toMove)
            move_evals = self.evaluate_moves(s,moves,network)
            if move_evals.numel() == 0:
                move_evals = torch.zeros(1) # would terminate prior so give zero value
            q_vals = move_evals
            if c == BLACK:
                q_vals *= -1.0
            out.append(q_vals)

        return out

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calc_loss(
            self,
            q_values,
            target_q_values,
            actions,
            rewards,
            done_mask
            ):

        gamma = self.config.gamma

        tmp = [qt.numel() for qt in target_q_values]
        if not all(tmp):
            for i in range(len(target_q_values)):
                if tmp[i] == 0:
                    target_q_values[i] = torch.zeros(1)

        q_targets = gamma * torch.tensor([torch.min(qt) for qt in target_q_values])
        q_targets = rewards + torch.bitwise_not(done_mask).to(torch.float32) * q_targets

        q_samples = [torch.index_select(qv,0,at) for qv,at in zip(q_values,actions)]
        q_samples = torch.cat(q_samples)
        result = F.mse_loss(q_samples,q_targets)
        return result
    
    # def build(self):
    #     if hasattr(self.config, "load_path"):
    #         print("Loading parameters from file:", self.config.load_path)
    #         load_path = Path(self.config.load_path)
    #         self.q_network.load_state_dict(torch.load(load_path, map_location="cpu"))
    #         print("Load successful!")
    #     else:
    #         print("Initializing parameters randomly")

    #         def init_weights(m):
    #             if hasattr(m, "weight"):
    #                 nn.init.xavier_uniform_(m.weight, gain=2** (1.0 / 2))
    #             if hasattr(m, "bias"):
    #                 nn.init.zeros(m.bias)

    #         self.q_network.apply(init_weights)
    #     self.q_network = self.q_network.to(self.device)
    #     self.target_network = self.target_network.to(self.device)
    #     self.add_optimizer()

    def update_step(self, t, replay_buffer, lr):
        s_batch, sp_batch, a_batch, r_batch, done_mask_batch = replay_buffer.sample(self.config.batch_size)
        
        done_mask_batch = done_mask_batch.bool()
        self.optimizer.zero_grad()
        color = [s.toMove for s in s_batch]
        q_values = self.get_q_values(s_batch, color, "q_network")
        with torch.no_grad():
            target_q_values = self.get_q_values(sp_batch, color, "target_network")
        loss = self.calc_loss(q_values, target_q_values,
                a_batch, r_batch, done_mask_batch)
        loss.backward()
        
        if self.config.grad_clip:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.config.clip_val
            ).item()
        else:
            total_norm = 0
        
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()

        return loss.item(), total_norm


    def save(self):
        torch.save(self.q_network.state_dict(), self.config.model_output)

    def train(self):
       
        rewards = deque(maxlen=self.config.num_episodes)
        losses = deque(maxlen=self.config.num_episodes)

        t = 0
        episode = 0
        
        while episode < self.config.num_episodes:
            episode += 1
            total_reward = 0
            total_loss = 0
            self.env.reset()

            legal_moves = self.env.allLegalMoves(self.env.toMove)
            self.env.makeMove(random.choice(legal_moves))
            legal_moves = self.env.allLegalMoves(self.env.toMove)
            self.env.makeMove(random.choice(legal_moves))

            m = 2

            print(f"episode = {episode}, t = {t}")
            
            while m < self.config.max_ep_len:
                m += 1
                t += 1

                if self.env.game_over:
                    break

                # choose action according to current Q and exploration
                legal_moves = self.env.allLegalMoves(self.env.toMove)
                best_move = self.choose_move(legal_moves)
                move = self.exp_schedule.get_action(best_move)

                # perform action in env
                state = board.Board(self.env)
                reward = self.calculate_reward(move)
                if self.env.toMove == BLACK: # need to negate reward for black
                    reward *= -1.0
                self.env.makeMove(move) # make the move
                done = self.env.game_over
                new_state = board.Board(self.env)
                action = legal_moves.index(move)

                # store the transition
                self.memory.add(
                    state,
                    new_state,
                    torch.Tensor([action]).long(),
                    torch.Tensor([reward]).float(),
                    torch.Tensor([done]).float(),
                )

                # # if there was a reward
                # if reward != 0:
                #     self.memory.add(
                #         state,
                #         new_state,
                #         torch.Tensor([action]).long(),
                #         torch.Tensor([reward]).float(),
                #         torch.Tensor([done]).float(),
                #     )

                # perform a training step
                loss_eval, grad_eval = self.train_step(
                    t, self.memory, self.config.learning_rate
                )
                total_loss += loss_eval

                # count reward
                total_reward += reward

            # updates to perform at the end of an episode
            rewards.append(total_reward)
            with open(self.config.output_path + "rewards.pkl", "wb") as f:
                pickle.dump(rewards, f)

            losses.append(total_loss / episode)
            with open(self.config.output_path + "losses.pkl", "wb") as f:
                pickle.dump(losses, f)

            print(self.env)

        with open(self.config.output_path + "rewards.pkl", "wb") as f:
            pickle.dump(rewards, f)
        print(rewards)


    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if t > self.config.learning_start and t % self.config.learning_freq == 0:
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)
            print(f"loss = {loss_eval}    grad = {grad_eval}")

        # occasionally update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target()

        # occasionally save the weights
        if t % self.config.saving_freq == 0:
            self.save()

        return loss_eval, grad_eval




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



def main():
    B = board.Board()

    engine = Agent(B,config.config_simpleagent())

    legal_moves = B.allLegalMoves(B.toMove)
    values = engine.evaluate_moves(B,legal_moves,"q_network")
    zipped = zip(legal_moves, values)
    for m,v in sorted(zipped, key=lambda x: x[1], reverse=True):
        print(f"eval({m}) = {v}")

    # make some moves first
    #B.makeMove("d2d4")
    #B.makeMove("e7e5")

    #print(B)
    #reward = engine.calculate_reward("d4xe5")
    #print(f"reward(d4xe5) = {reward}")
    
    B.makeMove("e2e4")
    B.makeMove("d7d5")
    #B.makeMove("e4xd5")

    print(B)

    legal_moves = B.allLegalMoves(B.toMove)
    values = engine.evaluate_moves(B,legal_moves,"q_network")
    zipped = zip(legal_moves, values)
    for m,v in sorted(zipped, key=lambda x: x[1], reverse=True):
        print(f"eval({m}) = {v}")

if __name__ == "__main__":
    main()

