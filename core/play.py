import gym

from core import utils
from core.agent import Agent
from core.settings import settings


def play():
    env = gym.make(settings.ENVIRONMENT)
    utils.set_seed(env)
    agent = Agent(
       env.observation_space.shape[0],
       env.action_space.shape[0],
       env.action_space.high)  # noqa
    if settings.RENDER:
        utils.render(env, init=True)
    for episode in range(settings.MAX_EPISODES):
        state = env.reset()
        episode_reward = 0.0
        t = 0
        for t in range(1, settings.MAX_TIMESTEPS + 1):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if settings.RENDER:
                utils.render(env)
            if done:
                break
        print(f"Reward: {episode_reward:.2f}, Timesteps: {t}")
    env.close()
