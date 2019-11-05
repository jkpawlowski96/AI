import app.ai.ai as ai

import time
import datetime
import numpy as np
import torch as t
import threading

class Model():
    inputs = None
    outputs = None
    date = None
    uid = None
    description = None
    online_learning = None
    batch_size=100

    active = True
    model = None

    batch=[]



    def __init__(self,inputs,outputs):
        self.date = str(datetime.datetime.now())

        self.inputs = inputs
        self.outputs = outputs

        self.model = ai.Model(self.inputs,self.outputs)

    def options(self, options):
        if 'online_learning' in options:
            self.online_learning = True
        else:
            self.online_learning = False

    def forward(self,x):
        x = self.to_tensor(x)
        x = self.model.forward(x.view((1,-1)))
        return self.from_tensor(x)
        
    def add(self,state,reward):
        state = self.to_tensor(state)
        reward = self.to_tensor(reward)

        self.batch.append((state,reward))
        if len(self.batch) >self.batch_size:
            self.train_on_batch()

    def train_on_batch(self):
        x = t.stack([t[0] for t in self.batch])
        y = t.stack([t[1] for t in self.batch])



        self.model.train(x,y)
        

    def to_tensor(self,x):
        x = [np.float(v) for v in x]
        x = t.FloatTensor(x)
        return x

    def from_tensor(self,x):
        resp = ""
        for v in x.view(-1):
            resp+=str(v.item())+" "
        resp = resp.replace(".",",")
        return resp
