import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import neptune

from DQN import DQN
from memory import *
from environment import *

import os
import numpy as np
#import pyautogui as pg


Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

def select_action(args, state):
    global steps_done

    board = state.reshape(1, args.width * args.height)
    unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]
    sample = random.random()
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end)* \
        math.exp(-1. * steps_done / args.eps_decay)
    steps_done += 1
    if args.neptune == 1:
        neptune.log_metric('Epsilon Decay', eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            moves = policy_net(state.to(device))
            moves[board!=-0.125] = torch.min(moves)
            action = torch.argmax(moves)
            return action
    else:
        move = torch.tensor(np.random.choice(unsolved)).to(device)
        return move

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def optimize_model(args):

    if len(memory) < 1000:
        return
    transitions = memory.sample(args.batch_size)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)


    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    #choose 1 action from policy net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(args.batch_size, device=device)
    
    if args.double == "True":
       
        q_tp1 = policy_net(non_final_next_states).detach()
        _, a_prime = q_tp1.max(1)
        q_target_tp1_values = target_net(non_final_next_states).detach()
        q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
        next_state_values[non_final_mask] = q_target_s_a_prime.squeeze()
        expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    else:
        next_state_values[non_final_mask] = torch.max(target_net(non_final_next_states),1)[0].detach()
        expected_state_action_values = (next_state_values * args.gamma) + reward_batch
    
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

    if args.neptune == 1:
        neptune.log_metric('MSE', loss.item())

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    my_lr_scheduler.step()
    

    if args.neptune == 1:
        neptune.log_metric('LR decay', my_lr_scheduler.get_last_lr()[0])


def train(args, env):
    progress_list, wins_list, ep_rewards = [], [], []
    for episode in range(args.episodes):
        env.reset()
        obs = env.state_im
        state = get_state(obs)
        total_reward = 0.0
        past_n_wins = env.n_wins
        for t in count():
            
            action = select_action(args, state)
            obs, reward, done = env.step(action,t)
            if t == 0 and done == True: #not consider 1 click loses.
                episode = episode - 1
                break
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            
            else:
                if reward > 0.9:
                    print(get_state(obs))
                    if args.neptune == 1:
                        neptune.log_metric('Clicks', t)
                next_state = None

            reward = torch.tensor([reward], device=device)
            memory.push(state, action, next_state, reward)
            state = next_state

            if steps_done > args.initial_memory: #we let the algorithm explore and put stuff in the memory before optimizing the model
                optimize_model(args)

                if steps_done % args.target_update == 0: 
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                if args.neptune == 1:
                        neptune.log_metric('Clicks_per_episode', t)
                break
        
        progress_list.append(env.n_progress) # n of non-guess moves
        ep_rewards.append(total_reward)
        if env.n_wins > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)
        
        if not episode % args.update_every:
            AGG_STATS_EVERY = args.update_every
            med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

            if args.neptune == 1:
                neptune.log_metric('progress med', med_progress)
                neptune.log_metric('win rate', win_rate)
                neptune.log_metric('med reward', med_reward)
            print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))


        if episode % args.save_model == 0:
            path = args.save_dir + '/' + str(args.conv_dim) +'_' + str(args.lin_dim) +'_' +str(args.batch_size)
            torch.save(policy_net, path +'_' +str(episode)+'.pth')

    #env.close()
    return

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

        if done:
            print("Finished Episode {} with reward {}".format(episode, total_reward))
            break
    return


if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--width', type=int, default=9, help='width of the board')
    parser.add_argument('--height', type=int, default=9, help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10, help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=50000, help='Number of episodes to train on')
    parser.add_argument('--name', type=str, default='', help='Name of model')
    parser.add_argument("--neptune", type=int, default=1)
    parser.add_argument("--nept_name", type=str, default='Minesweeper')
    parser.add_argument("--update_every", type=int, default=100)
    parser.add_argument("--save_model", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--conv_dim", type=int, default=128)
    parser.add_argument('--lin_dim', type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--eps_start", type=int, default=1)
    parser.add_argument("--eps_end", type=float, default=0.02)
    parser.add_argument("--eps_decay", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--double", type=str, default="True")
    parser.add_argument("--lr_decay", type=float, default=0.9999999)
    parser.add_argument("--initial_memory", type=int, default=50000)
    parser.add_argument("--target_update", type=int, default=5)
    parser.add_argument("--resume", type=str, default="False")
    parser.add_argument("--win", type=float, default=1)
    parser.add_argument('--lose', type=float, default=-1)
    parser.add_argument("--guess", type=float, default=-0.3)
    parser.add_argument("--prog", type=float, default=0.3)
    parser.add_argument('--no_prog', type=float, default=-0.9)
    parser.add_argument("--cum_prog", type=float, default=0.5)

    args = parser.parse_args()
    
    result_path = args.save_dir + '/'+ args.name + str(args.conv_dim) +'_' + str(args.lin_dim) + str(args.batch_size)

    pkl.dump(args, open(result_path +'.args.pkl', 'wb'), -1)

    print(args)
    PARAM = vars(args)
    exp_name = args.nept_name
    if args.neptune == 1:
        neptune.init('danastalyn/Mines')
        neptune.create_experiment(name=exp_name, params=PARAM)

    MEMORY_SIZE = args.initial_memory

    # create networks
    
    if args.resume == "False":
        policy_net = DQN(args).to(device)
    else: 
        policy_net = torch.load(result_path + '.pth')
        print('resume training: ' , result_path)
    target_net = DQN(args).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_decay)
    steps_done = 0
    
    # rewards
    rewards={'win':args.win, 'lose':args.lose, 'progress':args.prog, 'guess':args.guess, 'no_progress':args.no_prog, 'cum_prog':args.cum_prog}
    
    # create environment
    env = MinesweeperEnv(args.width, args.height, args.n_mines, rewards)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    train(args, env)
    print('Training complete')
    torch.save(policy_net, result_path + '.pth')
    
