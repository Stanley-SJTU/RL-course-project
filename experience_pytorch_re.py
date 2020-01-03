import numpy as np
import torch
import random


class Experience(object):
    
    def __init__(self, max_memory=100, discount=0.95):
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = 4   # set it manually as 4

    def remember(self, episode):
        # episode = [state, action, reward, state_next, game_over]
        # memory[i] = episode
        # state == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_data(self, data_size=10):
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        return random.sample(self.memory, data_size)
