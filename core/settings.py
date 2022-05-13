import json
import pathlib
from datetime import datetime, timezone
from typing import NamedTuple

import torch.cuda


class EnvInfo(NamedTuple):
    average_reward_to_solve: float
    max_timesteps: int
    fps: int


env_info = {
    "BipedalWalker-v3": EnvInfo(295.0, 1600, 30),
    "HopperBulletEnv-v0": EnvInfo(1000.0, 1000, 30),
    "Walker2DBulletEnv-v0": EnvInfo(1000.0, 1000, 30),
    "HalfCheetahBulletEnv-v0": EnvInfo(1000.0, 1000, 1000),
    "AntBulletEnv-v0": EnvInfo(1000.0, 1000, 30),
    "HumanoidBulletEnv-v0": EnvInfo(1000.0, 1000, 50),
    "AtlasPyBulletEnv-v0": EnvInfo(1000.0, 1000, 30),
}


class _Settings:
    MODE = None

    ENVIRONMENT = None
    PYBULLET = False
    AVERAGE_REWARD_TO_SOLVE = None
    MAX_EPISODES = 10_000
    MAX_TIMESTEPS = None
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 100

    START_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    NAME = None
    FOLDER = None
    CHECKPOINT = None
    RENDER = False
    FPS = None
    SECONDS_PER_FRAME = None
    SEED = None
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MEMORY_SIZE = 1_000_000
    BATCH_SIZE = 256
    INITIAL_LEARNING = 10_000
    GAMMA = 0.99
    LEARNING_RATE = 3e-4
    TAU = 0.005
    REWARD_SCALE = 2.0

    def apply_args(self, args):
        for key, value in vars(args).items():
            if value is NotSet:
                continue
            setting_name = key.upper()
            if not hasattr(self, setting_name):
                raise KeyError(f"Setting {setting_name} does not exist.")
            setattr(self, setting_name, value)

    def evaluate(self):
        if not isinstance(self.DEVICE, torch.device):
            self.DEVICE = torch.device(self.DEVICE)
        if self.CHECKPOINT is not None \
                and not isinstance(self.CHECKPOINT, pathlib.Path):
            self.CHECKPOINT = pathlib.Path(self.CHECKPOINT)
        self.NAME = self.START_TIME + "_" + self.ENVIRONMENT
        self.FOLDER = pathlib.Path("results") / self.NAME
        if self.AVERAGE_REWARD_TO_SOLVE is None:
            self.AVERAGE_REWARD_TO_SOLVE = \
                env_info[self.ENVIRONMENT].average_reward_to_solve
        if self.MAX_TIMESTEPS is None:
            self.MAX_TIMESTEPS = env_info[self.ENVIRONMENT].max_timesteps
        if self.FPS is None:
            self.FPS = env_info[self.ENVIRONMENT].fps
        self.SECONDS_PER_FRAME = 1 / self.FPS

        # Register PyBullet and PyBulletGym environments with OpenAI Gym.
        import pybullet_envs  # noqa
        import pybulletgym  # noqa
        env = "- " + self.ENVIRONMENT
        if env in pybullet_envs.getList() or env in pybulletgym.envs.get_list():
            self.PYBULLET = True

        self.validate()

    def validate(self):
        if self.DEVICE == torch.device('cuda') \
                and not torch.cuda.is_available():
            raise SystemExit(
                "Error: Device set to 'cuda', but it is not available. If you "
                "are using a checkpoint trained elsewhere, try using "
                "'--device cpu'.")

    def json(self):
        json_dict = {}
        for key in dir(self):
            if key.isupper() and not key.startswith("_"):
                value = getattr(self, key)
                if isinstance(value, pathlib.Path):
                    value = str(value)
                elif isinstance(value, torch.device):
                    value = str(value)
                json_dict[key] = value
        return json_dict

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "settings.json", 'w') as f:
            json.dump(self.json(), f)

    def load(self, path):
        with open(path / "settings.json", 'r') as f:
            json_dict = json.load(f)
        for key, value in json_dict.items():
            setattr(self, key, value)


class NotSet:
    pass


settings = _Settings()
