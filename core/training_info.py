import datetime
import json
import time

import matplotlib.pyplot as plt

from core.settings import settings


def _moving_average(array, window):
    if window % 2 != 1:
        raise ValueError("Window must be an odd number.")
    window = (window - 1) // 2
    length = len(array)
    averages = []
    for i in range(length):
        start = max(i - window, 0)
        end = min(i + 1 + window, length)
        list_slice = array[start:end]
        average = sum(list_slice) / len(list_slice)
        averages.append(average)
    return averages


class TrainingInfo:
    def __init__(self):
        self.start_time = time.time()
        self.running_time = 0
        self.episode = 1
        self.timestep = 1
        self.timesteps_total = 0
        self.episode_reward = 0.0
        self.episode_rewards = []
        self.episode_lengths = []

        if settings.CHECKPOINT is not None:
            self.load(settings.CHECKPOINT)

    def episode_finished(self):
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.timestep)
        self.timesteps_total += self.timestep

    def running_average(self, window):
        list_slice = self.episode_rewards[-window:]
        return sum(list_slice) / len(list_slice)

    def json(self):
        return {
            'running_time': self.running_time + time.time() - self.start_time,
            'episode': self.episode,
            'timesteps_total': self.timesteps_total,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths}

    def save(self):
        folder = settings.FOLDER / f"ep{self.episode}"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "training_info.json", 'w') as f:
            json.dump(self.json(), f)
        self.plot()
        settings.save(folder)

    def load(self, path):
        with open(path / "training_info.json", 'r') as f:
            json_dict = json.load(f)
        for key, value in json_dict.items():
            setattr(self, key, value)

    def plot(self):
        folder = settings.FOLDER / f"ep{self.episode}"
        folder.mkdir(parents=True, exist_ok=True)
        smoothed_rewards = _moving_average(self.episode_rewards, 11)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        xs = list(range(1, len(self.episode_rewards) + 1))
        plt.plot(xs, self.episode_rewards, 'lightgrey', label="Raw reward")
        plt.plot(xs, smoothed_rewards, 'tab:blue', label="Smoothed reward")
        plt.legend()
        plt.savefig(folder / "rewards.png")
        plt.close()

    def log(self, start=False, solved=False, newline=False, ended=False):
        """Output to terminal."""
        timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if start:
            print(f"{timestr} Training...")
            print(f"    Folder: {settings.FOLDER}")
            print(f"    Env: {settings.ENVIRONMENT}")
            print(f"    Average reward to solve: {settings.AVERAGE_REWARD_TO_SOLVE}")
            print(f"    Checkpoint: {settings.CHECKPOINT}")
            print(f"    Max episodes: {settings.MAX_EPISODES}, Max timesteps: {settings.MAX_TIMESTEPS}")
            print(f"    Initial learning: {settings.INITIAL_LEARNING}")
            print(f"    Device: {settings.DEVICE}, Seed: {settings.SEED}")
            print(f"    Learning rate: {settings.LEARNING_RATE}")
            print(f"    Gamma: {settings.GAMMA}, Tau: {settings.TAU}")
            print(f"    Reward scale: {settings.REWARD_SCALE}")
        elif solved:
            print(
                f"{timestr} Environment solved after {self.episode} episodes! "
                f"Running reward is now {self.running_average(100):.2f} and "
                f"the last episode received {self.episode_reward}!")
        elif ended:
            folder = settings.FOLDER / f"ep{self.episode}"
            print(f"{timestr} Training ended. Continue with checkpoint: {folder}")
        else:
            end = "\n" if newline else "\r"
            print(
                f"{timestr} "
                f"Episode {self.episode}  "
                f"Average: {self.running_average(100):>8.2f}   "
                f"Reward: {self.episode_reward:>8.2f}   "
                f"Timesteps: {self.timestep:>4}", end=end)
