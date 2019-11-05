import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sys


class AI(nn.Module):
    GAMMA = .999
    lr=0.001
    opt = 'Adam'

    layers = [1]

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
    
        expected_action = (action * self.GAMMA) + reward*1000
        
        #loss = F.smooth_l1_loss(action,expected_action)
        loss = F.mse_loss(action,expected_action)
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
        self.out = nn.Linear(layers[-1],outputs)

        self.update_optimizer()

    def forward(self, x):
        for i in range(self.depth):
            x = self.linear[i](x)
            x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        return x


class Model_simply(nn.Module):
    
    def __init__(self,inputs,outputs):
        super().__init__()
        
        self.GAMMA = .999
        
        self.l1 = nn.Linear(inputs,500)
        self.l2 = nn.Linear(500,100)
        self.out = nn.Linear(100,outputs)

        self.opt = t.optim.Adam(self.parameters(),lr=.01)


    def forward(self,x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.out(x)
        x = F.sigmoid(x)
        return x

    def train(self, state, reward):
        
        self.opt.zero_grad()
    
        action = self.forward(state)
    
        expected_action = (action * self.GAMMA) + reward*1000
        
        #loss = F.smooth_l1_loss(action,expected_action)
        loss = nn.MSELoss(action,expected_action)

        
        print(f'sum loss {loss.sum()}',file=sys.stderr)

        loss.backward()
        self.opt.step()

        return loss.sum()
        


