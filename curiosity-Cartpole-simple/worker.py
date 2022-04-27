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
import torchvision.transforms as TT



class Image:
    def __init__(self, env_name):
        self.env = env_name
        self.repeat = 4

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
        # self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        # self.image_memory[0, :, :] = img_rgb_resized
        # self.image_memory = img_rgb_resized

        # self.stack = collections.deque(maxlen=repeat)
        return img_rgb_resized

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

'''
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


resize = TT.Compose([TT.ToPILImage(),
                    TT.Resize(40, interpolation=Image.CUBIC),
                    TT.Grayscale(),
                    TT.ToTensor()])


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


class Image:
    def __init__(self, env_name):
        self.env = env_name
        self.repeat =4


    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array')
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(self.env, screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)

        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        img_rgb_resized = cv2.resize(screen, (42, 42), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255
        # Convert to float, rescale, convert to torch tensor

        # Resize, and add a batch dimension (BCHW)
        plt.imshow(img_rgb_resized)
        plt.show()
        return img_rgb_resized

    def reset(self):
        self.env.reset()
        for i in range(self.repeat):
            state = self.get_screen()
        return state

    def step(self, action):
        reward = 0.0
        done = False
        for i in range(self.repeat):
            next_state, reward, done, info = self.env.step(action)
            next_state = self.get_screen()
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

        return np.array(self.stack)'''


def worker(name, input_shape, n_actions, global_agent,
           optimizer, env_id, n_threads, global_idx, global_icm,
           icm_optimizer, icm):
    # frame_buffer = [input_shape[1], input_shape[2], 1]
    # env = make(env_id, shape=frame_buffer)
    # env = Image(env_id)
    # print(env)

    # env1 = Image(env_id)
    # env = StackFrames(env=env1)
    env = gym.make(env_id)
    env = Image(env)
    env = StackFrames(env)

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

    episode, max_steps, t_steps, scores = 0, 16000, 0, []
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
        # env.render(close = True)
        # wandb.log({'episode_score': score})
        # with global_idx.get_lock():
        #    global_idx.value += 1
        if name == '1':
            loss_i = T.sum(L_I)
            l_i.append(loss_i.detach().numpy())
            loss_f = T.sum(L_F)
            l_f.append(loss_f.detach().numpy())
            b = T.sum(loss)
            l.append(b.detach().numpy())
            a = T.sum(intrinsic_reward)
            intr.append(a.detach().numpy())  # for plotting intrinsic reward
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_score_5000 = np.mean(scores[max(0, episode - 5000): episode + 1])
            print('ICM episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'avg score (100) {:.2f}' ' intrinsic: {:.8f}' .format(
                episode, name, n_threads,
                t_steps / 1e6, score,
                avg_score, T.sum(intrinsic_reward)))
    if name == '1':
        x = [z for z in range(episode)]
        # plot_learning_curve(x, scores, 'Cartpole_pixels_ICM.png')
        np.savetxt("ICM_cartpole_pixels_score4.csv",
                   scores,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("ICM_cartpole_pixels_intr4.csv",
                   intr,
                   delimiter=",",
                   fmt='% s')

        np.savetxt("ICM_L_F_cartpole_pixels4.csv",
                   l_f,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("ICM_L_I_cartpole_pixels4.csv",
                   l_i,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("ICM_ON_LOSS_cartpole_pixels4.csv",
                   l,
                   delimiter=",",
                   fmt='% s')
        '''np.savetxt("ICM_cartpole_pixels_score3.csv",
                   scores,
                   delimiter=",",
                   fmt='% s')
        np.savetxt("ICM_ON_LOSS_cartpole_pixels3.csv",
                   l,
                   delimiter=",",
                   fmt='% s')'''



