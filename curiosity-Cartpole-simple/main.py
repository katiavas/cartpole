import os
import torch.multiprocessing as mp
from torch.backends import cudnn

from parallel_env import ParallelEnv
import gym
import random
import torch as T
import numpy as np

SEED = 111
os.environ['OMP_NUM_THREADS'] = '1'


if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    T.manual_seed(SEED)
    T.cuda.manual_seed(SEED)
    cudnn.deterministic = True
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    T.backends.cudnn.enabled = False
    mp.set_start_method('spawn')
    global_ep = mp.Value('i', 0)
    env_id = 'CartPole-v1'
    n_threads = 12
    n_actions = 2
    # input_shape = [4]
    input_shape = [4, 90, 40]
    # input_shape = [4, 40, 90]
    ICM = False
    # wandb.run.name = env_id+'/'+str(SEED) + '/ICM='+str(ICM)
    env = ParallelEnv(env_id=env_id, num_threads=n_threads,
                      n_actions=n_actions, global_idx=global_ep,
                      input_shape=input_shape, icm=True)
                      
# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
'''the state-space of the Cart-Pole has four dimensions of continuous values 
and the action-space has one dimension of two discrete values'''
