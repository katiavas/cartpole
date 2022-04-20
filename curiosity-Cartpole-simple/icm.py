import torch as T
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
# self.inverse = nn.Linear(6*2, 256)
# self.dense1 = nn.Linear(6+1, 256)
# self.new_state = nn.Linear(256, *input_dims)
from torch.backends import cudnn

'''In the inverse model you want to predict the action the agent took to cause this state to transition from time t to t+1
So you are comparing an integer vs an actual label/ the actual action the agent took
Multi-class classification problem
This is a cross entropy loss between the predicted action and the actual action the agent took'''
"The loss for the forward model is the mse between the predicted state at time t+1 and the actua state at time t+1  "
"So we have two losses : one that comes from the inverse model and one that comes from the forward model "
# alpha= 1 or 0.1
class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=2, alpha=1, beta=0.2):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.phi = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.inverse = nn.Linear(288*2, 256)
        self.pi_logits = nn.Linear(256, n_actions)

        self.dense1 = nn.Linear(288+1, 256)
        self.phi_hat_new = nn.Linear(256, 288)

        device = T.device('cpu')
        self.to(device)

    def forward(self, state, new_state, action):
        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(conv))
        conv = F.elu(self.conv3(conv))
        phi = self.phi(conv)

        conv_new = F.elu(self.conv1(new_state))
        conv_new = F.elu(self.conv2(conv_new))
        conv_new = F.elu(self.conv3(conv_new))
        phi_new = self.phi(conv_new)

        # [T, 32, 3, 3] to [T, 288]
        phi = phi.view(phi.size()[0], -1).to(T.float)
        phi_new = phi_new.view(phi_new.size()[0], -1).to(T.float)

        inverse = self.inverse(T.cat([phi, phi_new], dim=1))
        pi_logits = self.pi_logits(inverse)

        # from [T] to [T, 1]
        action = action.reshape((action.size()[0], 1))
        forward_input = T.cat([phi, action], dim=1)
        dense = self.dense1(forward_input)
        phi_hat_new = self.phi_hat_new(dense)

        return phi_new, pi_logits, phi_hat_new

    def calc_loss(self, states, new_states, actions):
        # don't need [] b/c these are lists of states
        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        new_states = T.tensor(new_states, dtype=T.float)

        phi_new, pi_logits, phi_hat_new = \
            self.forward(states, new_states, actions)

        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1 - self.beta) * inverse_loss(pi_logits, actions.to(T.long))

        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(phi_hat_new, phi_new)

        intrinsic_reward = self.alpha*0.5*((phi_hat_new-phi_new).pow(2)).mean(dim=1)
        return intrinsic_reward, L_I, L_F