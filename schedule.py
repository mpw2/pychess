
import random
import numpy as np

class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t: int):
        self.epsilon = self.eps_begin \
            + (float(t) / self.nsteps) * (self.eps_end - self.eps_begin)
        if t > self.nsteps:
            self.epsilon = self.eps_end

class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action: int) -> int:

        action = best_action

        if random.random() < self.epsilon:
            moves = self.env.allLegalMoves(self.env.toMove)
            action = random.choice(moves)

        return action
