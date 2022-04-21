import gym
import numpy as np
import torch as T
from torch.backends import cudnn
import cv2
from actor_critic import ActorCritic
from icm import ICM
from memory import Memory
import torch as T
import random
import collections
from wrapper import make


class Image:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.ROWS = 84
        self.COLS = 84
        self.repeat = 4
        self.image_memory = np.zeros((self.repeat, self.ROWS, self.COLS))

    def get_image(self):
        frame = self.env.render(mode='rgb_array')
        # convert an image from one colour space to another(from rgb to gray)
        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(new_frame, (84, 84), interpolation=cv2.INTER_CUBIC)
        # make all pixels black
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255
        # make pixel values between 0 and 1
        #  Every time before adding an image to our image_memory, we need to push our data by 1 frame, similar to deq
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0, :, :] = img_rgb_resized
        # self.image_memory = img_rgb_resized

        # self.stack = collections.deque(maxlen=repeat)
        return self.image_memory

    def reset(self):
        self.env.reset()
        for i in range(self.repeat):
            state = self.get_image()
        return state

    def step(self, action):
        reward = 0.0
        done = False
        for i in range(self.repeat):
            next_state, reward, done, info = self.env.step(action)
            next_state = self.get_image()
            reward += reward
            if done:
                # obs = self.env.reset()
                break
        return next_state, reward, done, info


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat=4):
        super(StackFrames, self).__init__(env)
        # Set our stack which will be a deque of maxlen repeat
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack)

def worker(name, input_shape, n_actions, global_agent,
           optimizer, env_id, n_threads, global_idx, global_icm,
           icm_optimizer, icm):


    # frame_buffer = [input_shape[1], input_shape[2], 1]
    # env = make(env_id, shape=frame_buffer)
    # env = Image(env_id)
    # print(env)
    '''env = gym.make(env_id)
    frame = env.render(mode='rgb_array')
    new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_rgb_resized = cv2.resize(new_frame, (42, 42), interpolation=cv2.INTER_CUBIC)
    # make all pixels black
    img_rgb_resized[img_rgb_resized < 255] = 0
    img_rgb_resized = img_rgb_resized / 255'''
    # env1 = Image(env_id)
    # env = StackFrames(env=env1)
    env = Image(env_id)

    T_MAX = 20

    local_agent = ActorCritic(input_shape, n_actions)

    if icm:
        local_icm = ICM(input_shape, n_actions)
        # T.save(local_icm.state_dict(), 'icm_weights.pth')
    else:
        local_icm = None
        intrinsic_reward = None

    memory = Memory()

    # frame_buffer = [input_shape[1], input_shape[2], 1]
    # env = make_atari(env_id, shape=frame_buffer)

    episode, max_steps, t_steps, scores = 0, 5000, 0, []
    intr = []
    l = []
    l_i = []
    l_f = []

    while episode < max_steps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            state = T.tensor([obs], dtype=T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, done, info = env.step(action)

            memory.remember(obs, action, obs_, reward, value, log_prob)
            score += reward
            obs = obs_
            ep_steps += 1
            t_steps += 1
            if ep_steps % T_MAX == 0 or done:
                states, actions, new_states, rewards, values, log_probs = \
                    memory.sample_memory()
                if icm:
                    intrinsic_reward, L_I, L_F = \
                        local_icm.calc_loss(states, new_states, actions)
                    # wandb.log({'forward_loss':L_F.item(), 'inverse_loss':L_I.item(), 'intrinsic_reward': intrinsic_reward})
                loss = local_agent.calc_loss(obs, hx, done, rewards,
                                             values, log_probs,
                                             intrinsic_reward)
                optimizer.zero_grad()
                hx = hx.detach_()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()
                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)
                for local_param, global_param in zip(
                        local_agent.parameters(),
                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                # local_agent.save('actor')
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                            local_icm.parameters(),
                            global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())

                memory.clear_memory()
        episode += 1
        # wandb.log({'episode_score': score})
        # with global_idx.get_lock():
        #    global_idx.value += 1
        if name == '1':

            loss_i = T.sum(L_I)
            l_i.append(loss_i.detach().numpy())
            # loss_f = T.sum(L_F)
            # l_f.append(loss_f.detach().numpy())
            b = T.sum(loss)
            l.append(b.detach().numpy())
            a = T.sum(intrinsic_reward)
            intr.append(a.detach().numpy())  # for plotting intrinsic reward
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_score_5000 = np.mean(scores[max(0, episode - 5000): episode + 1])
            print('ICM episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'avg score (100) {:.2f}'.format(
                episode, name, n_threads,
                t_steps / 1e6, score,
                avg_score))
    if name == '1':
        x = [z for z in range(episode)]
        # plot_learning_curve(x, scores, 'Cartpole_pixels_ICM.png')
        np.savetxt("cartpole_pixels_score1.csv",
                   scores,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("cartpole_pixels_intr1.csv",
                   intr,
                   delimiter=",",
                   fmt='% s')

        np.savetxt("L_I_cartpole_pixels1.csv",
                   l_i,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("ICM_ON_LOSS_cartpole_pixels1.csv",
                   l,
                   delimiter=",",
                   fmt='% s')
        # plot_learning_curve_with_shaded_error(x, scores, 'ICM_shaded_error_5000.png')
