import random
import time

import gym
import numpy as np
import torch

from core.settings import settings


def set_seed(env: gym.Env):
    if settings.SEED is not None:
        if settings.PYBULLET:
            env.seed(settings.SEED)
        else:
            env.reset(seed=settings.SEED)
        torch.manual_seed(settings.SEED)
        np.random.seed(settings.SEED)
        random.seed(settings.SEED)


def render(env, init=False):
    if init:
        if settings.PYBULLET:
            env.render(mode='human')
    else:
        if settings.PYBULLET:
            time.sleep(settings.SECONDS_PER_FRAME)
        else:
            env.render(mode='human')
