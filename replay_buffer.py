from collections import namedtuple
import random
import torch

transition = namedtuple("transition","state, next_state, action, reward, is_terminal")

class ReplayBuffer:
    # From MinAtar
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        batch_samples = transition(*zip(*samples))
        states = torch.cat(batch_samples.state)
        next_states = torch.cat(batch_samples.next_state)
        actions = torch.cat(batch_samples.action)
        rewards = torch.cat(batch_samples.reward).flatten()
        dones = torch.cat(batch_samples.is_terminal).flatten()

        return states, next_states, actions, rewards, dones
