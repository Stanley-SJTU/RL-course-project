import matplotlib.pyplot as plt
import math
import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchsnooper
from collections import namedtuple
from gym_maze.envs.maze_env import *
from skimage.transform import resize
from skimage.color import rgb2gray


# @torchsnooper.snoop()


def render_result():
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


def get_screen():
    envstate = env.render()
    envstate = rgb2gray(resize(envstate, (60, 60)))
    # envstate = envstate.transpose((2, 0, 1))
    # transform into torch
    envstate = torch.from_numpy(envstate.astype(np.float32))
    # add a dimension as to match the input of the network
    envstate = envstate.unsqueeze(0)

    return envstate.unsqueeze(0).to(device)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Experience(object):

    def __init__(self, max_memory_e=10000, discount_e=0.95):
        self.max_memory = max_memory_e
        self.discount = discount_e
        self.memory = list()
        self.num_actions = 4  # set it manually as 4

    def remember(self, episode_e):
        # episode = [state, action, reward, state_next, game_over]
        # memory[i] = episode
        # state == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode_e)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_data(self, data_size=10):
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        return random.sample(self.memory, data_size)


def select_action(state):
    global steps_done
    sample = random.random()
    if n_episode <= 1:
        eps_threshold = 0.7 + (EPS_START - 0.7) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
    elif n_episode <= 10:
        eps_threshold = 0.1 + (EPS_START - 0.1) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
    else:
        eps_threshold = 0.05 + (EPS_START - 0.05) * \
                        math.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1
    if n_episodes % 5000 == 0:
        print(eps_threshold, " at ", steps_done)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)


def optimize_model():
    if len(experience.memory) < BATCH_SIZE:
        return

    # Train neural network model
    episode_batch = experience.get_data(data_size=BATCH_SIZE)

    batch = Transition(*zip(*episode_batch))

    # Compute a mask of non-final states and concatenate the batch elements
    # This part is implemented from the cartpole game, it is not necessary in our case. I keep it
    # as not to modify too many codes
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    Q_sa = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # make sure the sample size is 128
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    targets = reward_batch + GAMMA * next_state_values

    loss = criterion(Q_sa, targets.unsqueeze(1))
    if t % 5000 == 0:
        print("loss is ", loss, " at ", t)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == "__main__":

    # Initialize the "maze" environment
    # env = MazeEnvRandom20x20Plus(enable_render=True)
    # env = MazeEnvSample10x10(enable_render=True)
    # env = MazeEnvSample30x30(enable_render=True)
    env = MazeEnvSample50x50(enable_render=True)
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
    NUM_EPISODES = 500
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 25
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = False
    REALTIME_RENDERING = False
    ENABLE_RECORDING = False
    max_memory = 10000

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 50
    TARGET_UPDATE = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    Creating a Q-Table for each state-action pair
    '''
    # q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    policy_net = DQN(screen_height, screen_width, NUM_ACTIONS).to(device)
    target_net = DQN(screen_height, screen_width, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    '''
    Begin simulation
    '''

    episode_durations = []

    num_streaks = 0
    env.render()

    # Initialize experience replay object
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    experience = Experience(max_memory_e=max_memory)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(policy_net.parameters())

    for n_episode in range(NUM_EPISODES):

        # Reset the environment
        env.reset()

        # the initial state
        total_reward = 0
        n_episodes = 0
        steps_done = 0

        envstate = get_screen()

        for t in range(MAX_T):

            # Select an action and execute
            action = select_action(envstate)
            obv, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            # Observe the result
            next_envstate = get_screen()

            # Store episode (experience)
            episode = [envstate, action, next_envstate, reward]
            experience.remember(episode)
            n_episodes += 1

            # Move to the next state
            envstate = next_envstate

            optimize_model()

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if REALTIME_RENDERING:
                time.sleep(0.1)

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps."
                      % (n_episode, t))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode timed out")

        # Update the target network
        if n_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # render_result()

