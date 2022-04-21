import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.distributions import Categorical
import random

class Encoder(nn.Module):

    def __init__(self, input_dims, feature_dim=288):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, (3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)

        shape = self.get_conv_out(input_dims)
        # Layer that will extract the features
        self.fc1 = nn.Linear(shape, feature_dim)

    def get_conv_out(self, input_dims):

        img = T.zeros(1, *input_dims)
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        shape = x.size()[0]*x.size()[1]*x.size()[2]*x.size()[3]
        # return int(np.prod(x.size()))
        return shape

    def forward(self, img):
        enc = F.elu(self.conv1(img))
        enc = F.elu(self.conv2(enc))
        enc = F.elu(self.conv3(enc))
        enc = F.elu(self.conv4(enc))

        enc_flatten = T.flatten(enc, start_dim=1)
        # enc_flatten = enc.view((enc.size()[0], -1))
        features = self.fc1(enc_flatten)

        return features


class ActorCritic(nn.Module):

    def __init__(self, input_dims , n_actions, gamma=0.99, tau=0.98, feature_dims=288):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.tau = tau
        self.encoder = Encoder(input_dims)

        # self.input = nn.Linear(*input_dims, 256)
        # self.dense = nn.Linear(256, 256)

        self.gru = nn.GRUCell(feature_dims, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def forward(self, img, hx):

        # x = F.relu(self.input(state))
        # x = F.relu(self.dense(x))
        # hx = self.gru(x, hx)
        state = self.encoder(img)
        hx = self.gru(state, hx)


        pi = self.pi(hx)
        v = self.v(hx)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], v, log_prob, hx

    def calc_R(self, done, rewards, values):
        values = T.cat(values).squeeze()
        if len(values.size()) == 1:  # batch of states
            R = values[-1] * (1-int(done))
        elif len(values.size()) == 0:  # single state
            R = values*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return,
                                dtype=T.float).reshape(values.size())
        return batch_return

    def calc_loss(self, new_states, hx, done,
                  rewards, values, log_probs, r_i_t=None):

        # if we are supplying an intrinsic reward them we want to add the reward from ICM
        if r_i_t is not None:
            # convert r_i_t to a numpy array because r_i_t is a tensor while rewards is a list of floating point values
            rewards += r_i_t.detach().numpy()
        returns = self.calc_R(done, rewards, values)
        # calculate generalised advantage
        # We need a value function for the state one step after our horizon
        # get the first element because other elements that the forward function returns are not the value function
        # (we want the element v )
        next_v = T.zeros(1, 1) if done else self.forward(T.tensor([new_states],
                                         dtype=T.float), hx)[1]

        values.append(next_v.detach())
        values = T.cat(values).squeeze()  # concatenate -> cat
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)
        #                   state of time at t+1  state of time at t
        delta_t = rewards + self.gamma*values[1:] - values[:-1]
        n_steps = len(delta_t)
        '''generalised advantage estimate : https://arxiv.org/pdf/1506.02438.pdf'''
        # There is gonna be an advantage for each time step in the sequence
        # So gae is gonna be a batch of states, T in length
        # So we have an advantage for each time step, which is proportional to a sum of all the rewards that follow
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k*delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)
        # Calculate losses
        actor_loss = -(log_probs*gae).sum()
        entropy_loss = (-log_probs*T.exp(log_probs)).sum()
        # [T] vs ()
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)  # mean squared error
        total_loss = actor_loss + critic_loss - 0.01*entropy_loss

        # total_loss = 0*actor_loss + 0*critic_loss - 0.01*entropy_loss*0
        return total_loss
