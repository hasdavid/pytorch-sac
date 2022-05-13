import gym

from core import utils
from core.agent import Agent
from core.settings import settings
from core.training_info import TrainingInfo


def training_loop(env, agent, info: TrainingInfo):
    info.log(start=True)
    if settings.RENDER:
        utils.render(env, init=True)
    for info.episode in range(info.episode, settings.MAX_EPISODES + 1):
        state = env.reset()
        info.episode_reward = 0.0
        for info.timestep in range(1, settings.MAX_TIMESTEPS + 1):
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            if settings.RENDER:
                env.render()
            agent.remember(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            info.episode_reward += reward
            if settings.RENDER:
                utils.render(env)
            if done:
                break
        info.episode_finished()
        if info.episode % settings.LOG_INTERVAL == 0:
            info.log(newline=True)
        info.log()
        if info.episode % settings.SAVE_INTERVAL == 0:
            agent.save(info.episode)
            info.save()
        if info.running_average(100) > settings.AVERAGE_REWARD_TO_SOLVE and info.episode >= 100:
            info.log(solved=True)
            break


def train():
    env = gym.make(settings.ENVIRONMENT)
    utils.set_seed(env)
    agent = Agent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.high)  # noqa
    info = TrainingInfo()
    try:
        training_loop(env, agent, info)
    except KeyboardInterrupt:
        pass
    agent.save(info.episode)
    info.save()
    info.log(ended=True)
    env.close()
