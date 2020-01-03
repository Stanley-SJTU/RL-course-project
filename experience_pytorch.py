import numpy as np
import torch


class Experience(object):
    
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model  # It is a cuda model
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

    def predict(self, state):
        state = torch.from_numpy(state.astype(np.float32))
        with torch.no_grad():
            predict_state = self.model(state)
        return predict_state

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]  # state 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
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
