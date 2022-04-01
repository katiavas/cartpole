import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
import gym

os.environ['OMP_NUM_THREADS'] = '1'


if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'CartPole-v1'
    n_threads = 12
    n_actions = 2
    input_shape = [4]
    env = ParallelEnv(env_id=env_id, n_threads=n_threads,
                      n_actions=n_actions, input_shape=input_shape, icm=True)
                      
                      
# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
'''the state-space of the Cart-Pole has four dimensions of continuous values 
and the action-space has one dimension of two discrete values'''
