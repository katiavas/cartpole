import random
import numpy as np
import torch as T

T.use_deterministic_algorithms(True)


class Memory:
    def __init__(self):
        self.seed = 111
        random.seed(self.seed)
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []

    def remember(self, state, action, reward, new_state, value, log_p):
        random.seed(self.seed)
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)
        self.new_states.append(new_state)
        self.log_probs.append(log_p)
        self.values.append(value)

    def clear_memory(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []

    def sample_memory(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        T.manual_seed(self.seed)

        return self.states, self.actions, self.rewards, self.new_states,\
               self.values, self.log_probs
