import torch.nn as nn

class config_simpleagent:
    def __init__(self):
        self.env_name = "SimpleAgent_max"
        self.training_iteration = 1
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

        #self.load_path = "results/SimpleAgent_max-1/model.weights"
        
        # model and training config
        self.saving_freq = 1000
        self.num_episodes = 1000 # number of episodes
        self.batch_size = 32 # number of moves in a batch
        self.buffer_size = 500 # number of moves in the replay buffer
        self.target_update_freq = 500
        self.max_ep_len = 200 # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 1.0 # the discount factor
        self.learning_freq = 4
        self.learning_start = 200

        self.eps_begin = 0.20
        self.eps_end = 0.01
        self.eps_nsteps = 10000
        
        # paramters for the value function model
        self.n_layers = 3
        self.layers_size = [64, 64, 64]

        # hyperparamters

def get_config(env_name):
    if env_name == "simpleagent":
        return config_simgpleagent()

