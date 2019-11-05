import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sys

class Model(nn.Module):
    
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
        
        loss = F.smooth_l1_loss(action,expected_action)
        #loss = nn.MSELoss(action,expected_action,reduction='none')

        
        print(f'sum loss {loss.sum()}',file=sys.stderr)

        loss.backward()
        self.opt.step()
