import random
from collections import deque

import numpy as np
import torch

from core.settings import settings


class ReplayBuffer:
    """Fixed-size replay buffer for storing experience."""

    def __init__(self, capacity):
        self._memory = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self._memory.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self._memory, k=batch_size)  # List of N 5-tuples.
        batch = zip(*batch)  # Returns N states, N actions, N rewards, ...
        states, actions, rewards, next_states, dones = batch  # Unpacking.

        states = torch.FloatTensor(np.vstack(states)).to(settings.DEVICE)
        actions = torch.FloatTensor(np.vstack(actions)).to(settings.DEVICE)
        rewards = torch.FloatTensor(np.vstack(rewards)).to(settings.DEVICE)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(settings.DEVICE)
        dones = torch.BoolTensor(np.vstack(dones)).to(settings.DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self._memory)
