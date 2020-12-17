## Torch Minesweeper

MineSweeper solver pytorch implementation using DQN and Double DQN

## Acknowledgements
* https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning
* https://github.com/jakejhansen/minesweeper_solver
* and Pytorch's tutorials on DQN
 
## Libraries
* Pytorch3.+ 
* pyautogui (for interacting with desktop)

## General Setup
There is no gym environment for MineSweeper, so it has to be builded from sratch (environment.py). It follows the same arrangement as in http://minesweeperonline.com/#beginner. 

* DQN agent: DQN.py
* Memory replay configuration: memory.py
* Main training: main.py
* Train from console: train.sh

## Arguments for training:

```
    parser.add_argument('--width', type=int, default=9, help='width of the board')
    parser.add_argument('--height', type=int, default=9, help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10, help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=50000, help='Number of episodes to train on')
    parser.add_argument('--name', type=str, default='', help='Name of model')
    parser.add_argument("--neptune", type=int, default=0)   #if 1, track variables in neptune.ai
    parser.add_argument("--nept_name", type=str, default='Minesweeper')  #name of experiment for neptune.ai
    parser.add_argument("--update_every", type=int, default=100)   #register values in neptune every [...]
    parser.add_argument("--save_model", type=int, default=10000)  #save checkpoint every [...]
    parser.add_argument("--save_dir", type=str, default='res_new') #directory to save model and checkpoints
    parser.add_argument('--batch_size', type=int, default=64)   #batch size for memory replay
    parser.add_argument("--conv_dim", type=int, default=128)   #dimension of conv layer
    parser.add_argument('--lin_dim', type=int, default=512)   #dimension of linear layer
    parser.add_argument("--gamma", type=float, default=0)   #discount factor
    parser.add_argument("--eps_start", type=int, default=1)   #max epsilon for epsilon greedy policy
    parser.add_argument("--eps_end", type=float, default=0.02)    #min epsilon for epsilon greedy policy
    parser.add_argument("--eps_decay", type=int, default=50000)    #epsilon decay rate
    parser.add_argument("--lr", type=float, default=1e-4)   #learning rate for agent
    parser.add_argument("--double", type=str, default="True")    #apply Doble DQN
    parser.add_argument("--lr_decay", type=float, default=0.9999999)    # decay rate of lr
    parser.add_argument("--initial_memory", type=int, default=50000)    #initial memory size
    parser.add_argument("--target_update", type=int, default=5)     #update target network every [...]
    parser.add_argument("--resume", type=str, default="False")   #to resume training
    parser.add_argument("--win", type=float, default=1)    #win reward
    parser.add_argument('--lose', type=float, default=-1)    #lose reward
    parser.add_argument("--guess", type=float, default==-0.3)   #guess reward
    parser.add_argument("--prog", type=float, default=0.3)    #progress reward
    parser.add_argument('--no_prog', type=float, default=-0.9)    #no progress reward
    parser.add_argument("--cum_prog", type=float, default=0.5)    #cumulative progress reward

```
To run, run in shell *python3.8 train.sh* adding the argument modifications of your liking.

## To test
For testing, it is a bit more complicated. Since we used our own environment file, we have to use the pyautogui library's screenshot function to get the images of the tiles from the online game http://minesweeperonline.com/#beginner. I provide my own screenshots in "mypics" directory, but I am positive that it will not work for other desktops, since I chose the screen size and a certain amount of pixels for my images.
So, the MinesweeperAgentWeb.py would recognize the tiles in the screen and use the actions from the trained model given in test.py. We have to control the amount of confidence of prediction for the tile recognition:

```
CONFIDENCES = {
    'unsolved': 0.99,
    'zero': 0.99,
    'one': 0.95,
    'two': 0.95,
    'three': 0.88,
    'four': 0.95,
    'five': 0.95,
    'six': 0.95,
    'seven': 0.95,
    'eight': 0.95
}
```

## Comments
The parameters as they are do not work well. The win rate after 300k episodes is barely 2%.
Feel free to explore on your own :)
