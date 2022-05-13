import argparse

import core.play
import core.train
from core.settings import settings, NotSet


parser = argparse.ArgumentParser(
    description="Implementation of the Soft Actor-Critic algorithm. Train the "
    "agent or watch it acting in the environment. You can stop the program "
    "gracefully with Ctrl+C. Training can be then continued using a "
    "checkpoint.")
parser.add_argument("mode", type=str, choices=['train', 'play'])
parser.add_argument("environment", type=str)
parser.add_argument("--checkpoint", type=str, default=NotSet, help=f"Load a "
                    f"model. Set it to a folder containing '.pt' file. "
                    f"Default: {settings.CHECKPOINT}")
parser.add_argument("--render", action='store_true', default=NotSet,
                    help=f"Render agent's struggle.")
parser.add_argument("--fps", type=int, default=NotSet, help=f"Rendering FPS. "
                    f"Only used when --render is set. Default depends on the "
                    f"environment. See env_info in settings.py.")
parser.add_argument("--seed", type=int, default=NotSet, help=f"Specify the "
                    f"seed. Note that Torch only guarantees reproducibility "
                    f"on the same version, machine and device. Default: "
                    f"{settings.SEED}")
parser.add_argument("--device", type=str, default=NotSet, help="Specify the "
                    "device to train on. Default: 'cuda:0' if available, else "
                    "'cpu'")
parser.add_argument("--average-reward-to-solve", type=float, default=NotSet,
                    help="Training will end when the average reward over the "
                    "last 100 episode reaches this value. Default depends on "
                    "the environment. See env_info in settings.py.")
parser.add_argument("--max-episodes", type=int, default=NotSet,
                    help=f"Maximum number of episodes to train. Default: "
                    f"{settings.MAX_EPISODES}")
parser.add_argument("--max-timesteps", type=int, default=NotSet,
                    help="Maximum number of timesteps in each episode. "
                    "Default depends on the environment. See env_info in "
                    "settings.py.")
parser.add_argument("--log-interval", type=int, default=NotSet, help=f"How "
                    f"often to keep the episode result in the terminal. "
                    f"Default: {settings.LOG_INTERVAL}")
parser.add_argument("--save-interval", type=int, default=NotSet, help=f"How "
                    f"often to save the model and current progress. "
                    f"Default: {settings.SAVE_INTERVAL}")
parser.add_argument("--learning-rate", type=float, default=NotSet,
                    help=f"Learning rate for the optimizer. Default: "
                    f"{settings.LEARNING_RATE}")
parser.add_argument("--reward-scale", type=float, default=NotSet,
                    help=f"Reward scaling as described in the SAC paper. "
                    f"Default: {settings.REWARD_SCALE}")


def main():
    args = parser.parse_args()
    settings.apply_args(args)
    settings.evaluate()
    if settings.MODE == 'train':
        core.train.train()
    elif settings.MODE == 'play':
        core.play.play()


if __name__ == '__main__':
    main()
