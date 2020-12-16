import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    
    def __init__(self, args):
        """
        Initialize Deep Q Network

        Args:
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.convdim = args.conv_dim 
        self.dense = args.lin_dim
        n_actions = args.width * args.height
        self.conv1 = nn.Conv2d(1, self.convdim , kernel_size=3)
        self.conv2 = nn.Conv2d(self.convdim, self.convdim, kernel_size=3)
        self.conv3 = nn.Conv2d(self.convdim, self.convdim, kernel_size=3)
        self.conv4 = nn.Conv2d(self.convdim, self.convdim, kernel_size=3)
        self.ln1 =  nn.Linear(self.convdim, self.dense)
        self.head = nn.Linear(self.dense, n_actions)
    
    def forward(self, x): #input (1,1,9,9)
        #print('DQN', x)
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.ln1(x.view(x.size(0), -1)))
        return self.head(x)


