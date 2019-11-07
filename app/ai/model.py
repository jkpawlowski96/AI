import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sys


class Model(nn.Module):
    GAMMA = .99
    lr=0.001
    opt = 'Adam'
    reward_max=0
    layers = [1]
    criterion = nn.MSELoss()

    def cross(self,other):
        state = self.state_dict()
        _state = other.state_dict()
        for k in state.keys():
            x = state[k]
            y = _state[k]
            state[k] = (x+y)/2
            

        self.load_state_dict(state)
        

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

    def train(self, state, action, reward):
        self.optimizer.zero_grad()
        action = self.forward(state)
        rmin = min(reward)
        rmax= max(reward)
        self.reward_max=max(rmax.item(),self.reward_max)       
        #expected_action = (action * self.GAMMA) + reward
        expected_action = action*self.GAMMA + (action/abs(action+.000000001))*reward
        loss = self.criterion(action,expected_action)
        loss.backward()
        self.optimizer.step()
        e = loss.item()
        print(f'sum loss {e}',file=sys.stderr)
        return e

class Model_deep(Model):
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

        self.cross(other=self)

    def forward(self, x):
        for i in range(self.depth):
            x = self.linear[i](x)
            x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        return x




