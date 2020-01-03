import matplotlib.pyplot as plt
import math
import os, sys, time, random
import numpy as np
import torch
import torchsnooper
from gym_maze.envs.maze_env import *
from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray


# @torchsnooper.snoop()
def simulate():

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = 0.9
    discount_factor = 0.99

    rewards = []

    num_streaks = 0
    env.render()

    # Initialize experience replay object
    experience = Experience(max_memory=max_memory)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy_net.parameters())

    for n_episode in range(NUM_EPISODES):

        loss = 0.0

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0
        n_episodes = 0
        envstate = get_screen()

        for t in range(MAX_T):

            # Select an action
            action = select_action(envstate, explore_rate)
            prev_envstate = envstate

            # execute the action
            obv, reward_neg, done, _ = env.step(action)
            reward_pos = 0.01 * obv[0] + 0.01 * obv[1]
            reward = reward_pos + reward_neg

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward
            envstate = get_screen()

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, env.is_game_over()]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size_e=data_size)

            optimizer.zero_grad()

            output = policy_net(inputs)
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()

            if t % 100 == 0:
                print("pos: ", reward_pos, "neg: ", reward_neg)

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if REALTIME_RENDERING:
                time.sleep(0.1)

            if env.is_game_over():
                sys.exit()

            if done:
                rewards.append(total_reward)
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (n_episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (n_episode, t, total_reward))

        # Print data
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d}"
        print(template.format(n_episode, NUM_EPISODES - 1, loss, n_episodes))

        # It's considered done when it's solved over 100 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(n_episode, n_episodes)
        learning_rate = get_learning_rate(n_episode)

        # Update the target network
        if n_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


def render_result(model_test):
    # Render tha maze
    env.render()

    # Reset the environment
    obv = env.reset()

    # the initial state
    envstate = env.render()
    envstate = resize(envstate, (10, 10))
    envstate = envstate.reshape((1, -1))
    prev_envstate = envstate
    total_reward = 0

    for t in range(MAX_T):

        # Select an action
        action = int(np.argmax(model_test.predict(prev_envstate)))

        # execute the action
        obv, reward, done, _ = env.step(action)

        # Observe the result
        envstate = env.render()
        envstate = resize(envstate, (10, 10))
        envstate = envstate.reshape((1, -1))
        total_reward += reward

        prev_envstate = envstate

        # Render tha maze
        env.render()
        time.sleep(0.2)

        if env.is_game_over():
            return

        if done:
            print("Result Episode is finished after {} time steps with total reward = {}".format(t, total_reward))
            return


class Experience(object):

    def __init__(self, max_memory=100, discount=0.95):
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = 4  # set it manually as 4

    def remember(self, episode):
        # episode = [state, action, reward, state_next, game_over]
        # memory[i] = episode
        # state == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, state):
        state = torch.from_numpy(state.astype(np.float32))
        with torch.no_grad():
            predict_state = target_net(state)
        return predict_state

    def get_data(self, data_size_e=10):
        env_size = self.memory[0][0].shape[1]  # state 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size_e = min(mem_size, data_size_e)
        inputs = np.zeros((data_size_e, env_size))
        targets = np.zeros((data_size_e, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size_e, replace=False)):
            state, action, reward, state_next, game_over = self.memory[j]
            inputs[i] = state
            # There should be no target values for actions not taken.
            targets[i] = self.predict(state).detach().numpy()
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(state_next).detach().numpy())
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa

        inputs = torch.from_numpy(inputs.astype(np.float32))
        targets = torch.from_numpy(targets.astype(np.float32))

        return inputs, targets


def get_screen():
    envstate = env.render()
    envstate = rgb2gray(resize(envstate, (30, 30)))
    envstate = envstate.reshape((1, -1))
    # envstate = envstate.transpose((2, 0, 1))
    # envstate = torch.from_numpy(envstate.astype(np.float32))

    return envstate


def build_model(img_maze):

    model = torch.nn.Sequential(
        torch.nn.Linear(img_maze.size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, NUM_ACTIONS)
    )

    return model


def select_action(state, explore_rate):
    # Select a random action
    state = torch.from_numpy(state.astype(np.float32))
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        with torch.no_grad():
            output = policy_net(state).detach().numpy()
            action = int(np.argmax(output))
    return action


def get_explore_rate(epoch, t):
    if epoch < 1:
        eps_threshold = 0.7 + (EPS_START - 0.7) * \
                        math.exp(-1. * t / EPS_DECAY)
    elif epoch < 10:
        eps_threshold = 0.1 + (EPS_START - 0.1) * \
                        math.exp(-1. * t / EPS_DECAY)
    else:
        eps_threshold = 0.05 + (0.1 - 0.05) * \
                        math.exp(-1. * t / EPS_DECAY)
    return eps_threshold


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":

    # Initialize the "maze" environment
    # env = MazeEnvRandom20x20Plus(enable_render=True)
    env = MazeEnvSample10x10(enable_render=True)
    # env = MazeEnvSample30x30(enable_render=True)
    # env = MazeEnvSample50x50(enable_render=True)
    # env = MazeEnvSample100x100(enable_render=True)
    # env = MazeEnvSample200x200(enable_render=True)
    '''
    Defining the environment related constants
    '''

    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 50000
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 60
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = False
    REALTIME_RENDERING = False
    ENABLE_RECORDING = False
    max_memory = 10000
    data_size = 50
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 50
    TARGET_UPDATE = 5

    '''
    Creating a Q-Table for each state-action pair
    '''
    # q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    env_maze = get_screen()
    policy_net = build_model(env_maze)
    target_net = build_model(env_maze)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    '''
    Begin simulation
    '''

    simulate()

    # render_result(model_maze)

