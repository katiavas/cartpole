import random
import numpy as np
import torch as T

# T.use_deterministic_algorithms(True)
from torch.backends import cudnn


class Memory:
    # T.use_deterministic_algorithms(True)

    def __init__(self):
        '''T.use_deterministic_algorithms(True)
        # self.seed = 111
        SEED = 111
        T.use_deterministic_algorithms(True)
        random.seed(SEED)
        np.random.seed(SEED)
        T.manual_seed(SEED)
        T.cuda.manual_seed(SEED)
        cudnn.deterministic = True
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.enabled = False'''
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []

    def remember(self, state, action, reward, new_state, value, log_p):
        # T.manual_seed(SEED)
        # T.use_deterministic_algorithms(True)
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        '''SEED = 111
        T.use_deterministic_algorithms(True)
        random.seed(SEED)
        np.random.seed(SEED)
        T.manual_seed(SEED)
        T.cuda.manual_seed(SEED)
        cudnn.deterministic = True
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.enabled = False'''

        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)
        self.new_states.append(new_state)
        self.log_probs.append(log_p)
        self.values.append(value)

    def clear_memory(self):
        # random.seed(self.seed)
         #np.random.seed(self.seed)
        # T.manual_seed(self.seed)
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []

    def sample_memory(self):
        # T.use_deterministic_algorithms(True)
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # T.manual_seed(self.seed)

        return self.states, self.actions, self.rewards, self.new_states,\
               self.values, self.log_probs
