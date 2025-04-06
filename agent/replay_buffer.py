import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.memory = deque(maxlen=capacity)  # auto-removes oldest when full
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
