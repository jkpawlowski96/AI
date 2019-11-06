import app.ai.ai as ai

import time
import datetime
import numpy as np
import torch as t
import threading

import plotly
import plotly.graph_objs as go

import json

class Model():
    inputs = None
    outputs = None
    date = None
    uid = None
    description = None
    online_learning = True
    batch_size=1000
    lr=0.001
    GAMMA=0.999
    opt='Adam'
    layers=[]
    active = True
    model = None

    epoch=0
    batch=[]
    losses=[]


    def __init__(self,inputs,outputs):
        self.date = str(datetime.datetime.now())

        self.inputs = inputs
        self.outputs = outputs

        #self.model = ai.Model(self.inputs,self.outputs)
        self.model = ai.Model_deep( self.inputs,
                                    self.outputs)

    def update_model(self,form=None):
        if form is not None:
            self.options(form.getlist('options'))
            self.lr_percent = form['lr_percent']
            self.lr = np.float(form['lr'])
            self.opt = form['opt']

            self.GAMMA = np.float(form['GAMMA'])
            self.batch_size = np.int(form['batch_size'])
            for n in range(len(self.layers)):
                try:
                    l = form['l'+str(n)]
                    l = np.int(l)
                    if l <=0:
                        self.layers = self.layers[:n-1]
                        pass
                    self.layers[n]=l
                except Exception:
                    self.layers = self.layers[:n-1]
                    pass

                    

        if self.layers is not self.model.layers:
            self.model = ai.Model_deep(self.inputs,self.outputs,
                                    layers=self.layers.copy())

        if self.lr is not self.model.lr:
            self.model.update_optimizer(lr=self.lr)
        
        if self.GAMMA is not self.model.GAMMA:
            self.model.GAMMA = self.GAMMA

        if self.opt is not self.model.opt:
            self.model.update_optimizer(opt = self.opt)
        
    def plot_losses(self):
        

        """
        data = [
            go.Bar(
                x=list(range(len(self.losses))), # assign x as the dataframe column 'x'
                y=np.array(self.losses)
            )
        ]
        """
        x=list(range(len(self.losses))) # assign x as the dataframe column 'x'
        y=np.array(self.losses)

        
        data=[
        go.Scatter(x=x, y=y,
                            mode='lines+markers',
                            name='lines+markers')
        ]

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

        

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
        if not self.online_learning:
            return None

        state = self.to_tensor(state)
        reward = self.to_tensor(reward)

        self.batch.append((state,reward))
        if len(self.batch) >self.batch_size:
            loss = self.train_on_batch()
            self.losses.append(loss)
            self.batch = []

    def train_on_batch(self):
        x = t.stack([t[0] for t in self.batch])
        y = t.stack([t[1] for t in self.batch])

        loss = self.model.train(x,y)
        return loss
        

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

    def n_layers(self):
        return len(self.layers)