import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sys


class AI(nn.Module):
    GAMMA = .99
    lr=0.001
    opt = 'Adam'
    reward_max=0
    layers = [1]
    criterion = nn.MSELoss()

    def update_optimizer(self,lr=None,opt=None):
        if lr is not None:
            self.lr = lr
        if opt is not None:
            self.opt = opt

        if self.opt is 'Adam':
            self.optimizer = t.optim.Adam(self.parameters(),lr=self.lr)
        elif self.opt is 'SGD':
            self.optimizer = t.optim.SGD(self.parameters(),lr=self.lr)
        else:
            self.optimizer = t.optim.Adam(self.parameters(),lr=self.lr)

    def train(self, state, reward):
        
        self.optimizer.zero_grad()
    
        action = self.forward(state)

        rmin = min(reward)
        rmax= max(reward)
        self.reward_max=max(rmax.item(),self.reward_max)
        #reward = reward - rmin # to [0;n]
        #reward = reward / reward.max() # to [0:1]
        #reward = reward *2 -1 # to [-1:1]        


        #expected_action = (action * self.GAMMA) + action*reward

        #expected_action = (action * self.GAMMA) + reward

        expected_action = action*self.GAMMA + (action/abs(action))*reward


        #loss = F.smooth_l1_loss(action,expected_action)
        #loss = F.mse_loss(action,expected_action)
        #loss = F.l1_loss(action,expected_action)
        loss = self.criterion(action,expected_action)
        e = loss.sum().item()
        print(f'sum loss {loss.sum()}',file=sys.stderr)

        loss.backward()
        self.optimizer.step()
        return e

class Model_deep(AI):
    def __init__(self, inputs, outputs,layers=[1]):
        super().__init__()

        self.layers = layers
        self.depth = len(layers)
        self.linear = []
        for l in layers:
            self.linear.append(nn.Linear(inputs, l))
            inputs=l

        self.linear = nn.ModuleList(self.linear)
        self.out = nn.Linear(inputs,outputs)

        self.update_optimizer()

    def forward(self, x):
        for i in range(self.depth):
            x = self.linear[i](x)
            x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        return x




