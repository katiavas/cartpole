import collections
import cv2
import numpy as np
import gym
import random
import torch as T


# https://livebook.manning.com/book/deep-reinforcement-learning-in-action/chapter-8/v-7/63
# Step the environment with the given action Repeat action, sum reward, and max over last observations
class RepeatAction(gym.Wrapper):
    def __init__(self, shape, env, repeat=4, fire_first=False):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat
        self.shape = shape
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env_step(1)
        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])

    def observation(self, obs):
        obs = self.env.render(mode = 'rgb_array')
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        print(new_obs.shape)

        return new_obs



class StackFrames(gym.ObservationWrapper):

    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack)

    def observation(self, observation):
        self.stack.append(observation)
        print(self.stack)

        return np.array(self.stack)


def make(env_name, shape=(84, 84, 1), repeat=4):
    env = gym.make(env_name)
    env = RepeatAction(shape=shape, env=env, repeat=repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env
