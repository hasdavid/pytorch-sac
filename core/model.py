import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim

from core.settings import settings


def _initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


class Actor(nn.Module):
    """Actor (policy) network that maps states to actions."""

    def __init__(self, state_size, action_size, action_max):
        super().__init__()
        self.reparam_noise = 1e-6
        self.learning_rate = settings.LEARNING_RATE
        self.action_max = torch.FloatTensor(action_max).to(settings.DEVICE)

        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),)
        self.means = nn.Linear(256, action_size)
        self.stds = nn.Linear(256, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.apply(_initialize_weights)
        self.to(settings.DEVICE)

    def forward(self, state):
        x = self.net(state)
        means = self.means(x)
        stds = self.stds(x)
        # We want to limit the standard deviation to a certain range. Instead
        # of clamp we could also use a sigmoid activation, but that is more
        # computationally expensive.
        stds = torch.clamp(stds, self.reparam_noise, 1)
        return means, stds

    def sample_action(self, state, reparametrize=True):
        means, stds = self(state)
        distribution = dist.Normal(means, stds)
        if reparametrize:
            actions = distribution.rsample()
        else:
            actions = distribution.sample()

        # Scale if the environment has different action range than -1 and 1.
        action = torch.tanh(actions) * self.action_max

        # Log_probs for calculation of loss function.
        log_probs = distribution.log_prob(actions)
        # We add the reparam_noise here to prevent log(0).
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        # We are taking the sum, because we need a scalar quantity for
        # calculation of our loss. The log_probs will have number of
        # components equal to number of actions.
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs


class Critic(nn.Module):
    """Critic (Q-value) network that maps (state, action) pairs to Q-values."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.learning_rate = settings.LEARNING_RATE
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.apply(_initialize_weights)
        self.to(settings.DEVICE)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class Value(nn.Module):
    """Value network returns values of states."""

    def __init__(self, state_size):
        super().__init__()
        self.learning_rate = settings.LEARNING_RATE
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.apply(_initialize_weights)
        self.to(settings.DEVICE)

    def forward(self, state):
        return self.net(state)
