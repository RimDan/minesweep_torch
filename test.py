import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import neptune

import time
from DQN import DQN
from memory import *
from environment import *

import os
import numpy as np
import pyautogui as pg
from MinesweeperAgentWeb import MinesweeperAgentWeb


def test(env, n_episodes, policy_net):
    pg.FAILSAFE = True
    agent = MinesweeperAgentWeb(policy_net)
    for episode in range(n_episodes):
        agent.reset()

        done = False
        while not done:
            current_state = agent.state
            action = agent.get_action(current_state)

            new_state, done = agent.step(action)


if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--width', type=int, default=9, help='width of the board')
    parser.add_argument('--height', type=int, default=9, help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10, help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument("--best_model", type=str, default='./results/min28.pth')
    parser.add_argument("--win", type=float, default=1)
    parser.add_argument('--lose', type=float, default=-1)
    parser.add_argument("--guess", type=float, default=-0.3)
    parser.add_argument("--prog", type=float, default=0.3)
    parser.add_argument('--no_prog', type=float, default=-0.9)
    parser.add_argument("--cum_prog", type=float, default=0.5)

    args = parser.parse_args()

    print(args)
     # rewards
    rewards={'win':args.win, 'lose':args.lose, 'progress':args.prog, 'guess':args.guess, 'no_progress':args.no_prog, 'cum_prog':args.cum_prog}

    env = MinesweeperEnv(args.width, args.height, args.n_mines, rewards)

    policy_net = torch.load(args.best_model)
    test(env, args.episodes, policy_net)
