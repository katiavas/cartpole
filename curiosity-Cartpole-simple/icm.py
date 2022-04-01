import torch as T
import torch.nn as nn
import torch.nn.functional as F

# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
# self.inverse = nn.Linear(6*2, 256)
# self.dense1 = nn.Linear(6+1, 256)
# self.new_state = nn.Linear(256, *input_dims)

'''In the inverse model you want to predict the action the agent took to cause this state to transition from time t to t+1
So you are comparing an integer vs an actual label/ the actual action the agent took
Multi-class classification problem
This is a cross entropy loss between the predicted action and the actual action the agent took'''
"The loss for the forward model is the mse between the predicted state at time t+1 and the actua state at time t+1  "
"So we have two losses : one that comes from the inverse model and one that comes from the forward model "


class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=2, alpha=1, beta=0.2):
        super(ICM, self).__init__()
        T.manual_seed(5)
        self.alpha = alpha
        self.beta = beta
        # hard coded for cartPole environment
        self.inverse = nn.Linear(4 * 2, 256)
        self.pi_logits = nn.Linear(256, n_actions)

        self.dense1 = nn.Linear(4 + 1, 256)
        self.new_state = nn.Linear(256, 4)

        device = T.device('cpu')
        self.to(device)

    # Forward model takes the action and the current state and predicts the next state
    def forward(self, state, new_state, action):
        """ We have to concatenate a state and action and pass it through the inverse layer """
        "and activate it with an elu activation--> exponential linear"
        # Create inverse layer
        inverse = F.elu(self.inverse(T.cat([state, new_state], dim=1)))
        pi_logits = self.pi_logits(inverse)

        # Forward model
        # from [T] to [T,1]
        action = action.reshape((action.size()[0], 1))
        # Activate the forward input and get a new state on the other end
        forward_input = T.cat([state, action], dim=1)
        dense = F.elu(self.dense1(forward_input))
        state_ = self.new_state(dense)

        return pi_logits, state_

    def calc_loss(self, state, new_state, action):
        state = T.tensor(state, dtype=T.float)
        action = T.tensor(action, dtype=T.float)
        new_state = T.tensor(new_state, dtype=T.float)
        # feed/pass state, new_state , action through our network
        pi_logits, state_ = self.forward(state, new_state, action)
        "Our inverse loss is a cross entropy loss because this will generally have more than two actions"
        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1 - self.beta) * inverse_loss(pi_logits, action.to(T.long))
        "Forward loss is mse between predicted new state and actual new state"
        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(state_, new_state)
        # dim=1 for mean(dim=1) is very important. If you take that out it will take the mean across all dimensions
        # and you just get a single number, which is not useful
        # because the curiosity reward is associated with each state, so you have to take the mean across that first
        # dimension which is the number of states
        intrinsic_reward = self.alpha * ((state_ - new_state).pow(2)).mean(dim=1)
        return intrinsic_reward, L_I, L_F
