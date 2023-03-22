import torch.nn as nn

from constants import *

class config_simpleagent_v2:
    def __init__(self,train_iter=None):
        self.env_name = "SimpleAgent_v2"
        self.training_iteration = 11
        self.record = False
        self.output_path = "results/{}-{}/".format(
            self.env_name, self.training_iteration
        )
        self.model_output = self.output_path + "model.weights"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        self.load_path = "results/SimpleAgent_v2-10/model.weights"
        if not train_iter is None:
            self.load_path = "results/SimpleAgent_v2-{}/model.weights".format(
                train_iter
            )
            self.training_iteration = train_iter + 1
        
        # model and training config
        self.saving_freq = 1000
        self.num_episodes = 500 # number of episodes
        self.batch_size = 32 # number of moves in a batch
        self.buffer_size = 1000 # number of moves in the replay buffer
        self.target_update_freq = 1000
        self.max_ep_len = 200 # maximum episode length
        self.learning_rate = 0.1
        self.gamma = 1.0 # the discount factor
        self.learning_freq = 10
        self.learning_start = 200

        self.sub_steps = 1

        self.grad_clip = True
        self.clip_val = 100

        self.eps_begin = 0.05
        self.eps_end = 0.01
        self.eps_nsteps = 10000
        
        # paramters for the value function model
        self.n_layers = 2
        self.layers_size = [64*NUM_PIECES, 64]

        # hyperparamters

class config_simpleagent:
    def __init__(self,train_iter=None):
        self.env_name = "SimpleAgent_v3"
        self.training_iteration = 2
        self.record = False
        self.output_path = "results/{}-{}/".format(
            self.env_name, self.training_iteration
        )
        self.model_output = self.output_path + "model.weights"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        self.load_path = "results/SimpleAgent_v3-1/model.weights"
        if not train_iter is None:
            self.load_path = "results/SimpleAgent_v3-{}/model.weights".format(
                train_iter
            )
            self.training_iteration = train_iter + 1
        
        # model and training config
        self.saving_freq = 1000
        self.num_episodes = 2000 # number of episodes
        self.batch_size = 32 # number of moves in a batch
        self.buffer_size = 1000 # number of moves in the replay buffer
        self.target_update_freq = 1000
        self.max_ep_len = 200 # maximum episode length
        self.learning_rate = 0.1
        self.gamma = 1.0 # the discount factor
        self.learning_freq = 10
        self.learning_start = 200

        self.sub_steps = 1

        self.grad_clip = True
        self.clip_val = 100

        self.eps_begin = 0.05
        self.eps_end = 0.01
        self.eps_nsteps = 10000
        
        # paramters for the value function model
        self.n_layers = 1
        self.layers_size = [64]

        # hyperparamters

def get_config(env_name):
    if env_name == "simpleagent":
        return config_simpleagent()

