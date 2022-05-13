# Soft Actor-Critic in PyTorch

Implementation of Soft Actor-Critic algorithm[[1]](#references) in PyTorch.

Author: David Hás 

## Installation

Program is tested on Ubuntu system with Python 3.9.

It is recommended to use a virtual environment. Training on the GPU requires
a CUDA capable system.

Install the requirements like this:

```sh
pip install -r requirements.txt
```

If your system cannot find some package, try to install its latest version
available to you. This way the program may run, but it is not guaranteed.

## Usage

To train the Soft Actor-Critic agent, use the following command:

```sh
python main.py train ENVIRONMENT \
    --checkpoint CHECKPOINT \
    --reward-scale REWARD_SCALE \
    --seed SEED
```

Watch the agent perform in the environment like this:
```sh
python main.py play ENVIRONMENT \
    --checkpoint CHECKPOINT \
    --render RENDER \
    --fps FPS
```

Run `python main.py --help` to see all available arguments with their
description. You can shut the program down gracefully using Ctrl+C. After that
you can restart the training from where you left off using a checkpoint.

All of the following environments will work with this program:

```
HopperBulletEnv-v0
Walker2DBulletEnv-v0
HalfCheetahBulletEnv-v0
AntBulletEnv-v0
HumanoidBulletEnv-v0
AtlasPyBulletEnv-v0
```

For example, to watch a trained Atlas robot, run the following:

```sh
python main.py play AtlasPyBulletEnv-v0 --render --checkpoint trained/AtlasPyBulletEnv-v0
```

Example of training Atlas:

```sh
python main.py train AtlasPyBulletEnv-v0
```

## References

[[1]](https://doi.org/10.48550/arXiv.1801.01290) HAARNOJA, Tuomas, Aurick ZHOU,
Pieter ABBEEL a Sergey LEVINE. Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor. 2018. DOI:
10.48550/arXiv.1801.01290
