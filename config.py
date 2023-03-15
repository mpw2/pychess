import torch.nn as nn

class config_simpleagent:
    def __init__(self):
        self.env_name = "SimpleAgent"
        self.record = False
        self.output_path = "results/{}/".format(
            self.env_name
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1
        
        # model and training config
        self.num_batches = 50 # number of batches trained on
        self.batch_size = 2000 # number of moves in a batch
        self.max_ep_len = 100 # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 1.0 # the discount factor
        
        # paramters for the value function model
        self.n_layers = 3
        self.layers_size = [64, 64, 64]

        # hyperparamters

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

def get_config(env_name):
    if env_name == "simpleagent":
        return config_simgpleagent()

