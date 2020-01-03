import matplotlib.pyplot as plt
import math
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from gym_maze.envs.maze_env import *
import experience as exp
from PIL import Image
from skimage.transform import resize


def simulate(model):

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = 0.1
    discount_factor = 0.99

    rewards = []

    num_streaks = 0
    env.render()

    # Initialize experience replay object
    experience = exp.Experience(model, max_memory=max_memory)

    for n_episode in range(NUM_EPISODES):

        loss = 0.0
        if n_episode > 20:
            explore_rate = 0.05

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0
        n_episodes = 0
        envstate = env.render()
        envstate = resize(envstate, (10, 10))
        envstate = envstate.reshape((1, -1))

        for t in range(MAX_T):

            prev_envstate = envstate
            # Get next action
            action = select_action(prev_envstate, model, explore_rate)

            # execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward
            envstate = env.render()
            envstate = resize(envstate, (10, 10))
            envstate = envstate.reshape((1, -1))

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, env.is_game_over()]
            experience.remember(episode)
            n_episodes += 1

            # Setting up for the next iteration
            state_0 = state

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)

            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=64,
                verbose=0,
            )

            loss = model.evaluate(inputs, targets, verbose=0)

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
        explore_rate = get_explore_rate(n_episode)
        learning_rate = get_learning_rate(n_episode)

    plt.plot(rewards)
    plt.title('Episode rewards')
    plt.xlabel('n_episode')
    plt.ylabel('Reward')
    plt.show()


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


def build_model(img_maze, lr=0.001):
    model = Sequential()
    model.add(Dense(img_maze.size, input_shape=(img_maze.size,)))
    model.add(PReLU())
    model.add(Dense(img_maze.size))
    model.add(PReLU())
    model.add(Dense(NUM_ACTIONS))
    model.compile(optimizer='adam', loss='mse')
    return model


def select_action(state, model_train, explore_rate):
    # Select a random action
    if np.random.rand() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(model_train.predict(state)))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


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
    NUM_EPISODES = 500
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = False
    REALTIME_RENDERING = False
    ENABLE_RECORDING = False
    max_memory = 100
    data_size = 50

    '''
    Creating a Q-Table for each state-action pair
    '''
    # q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    env_maze = env.render()
    env_maze = resize(env_maze, (10, 10))
    env_maze = env_maze.reshape((1, -1))
    model_maze = build_model(env_maze)

    '''
    Begin simulation
    '''
    # recording_folder = "/tmp/maze_q_learning"

    # if ENABLE_RECORDING:
    #     env.monitor.start(recording_folder, force=True)

    simulate(model_maze)

    render_result(model_maze)

