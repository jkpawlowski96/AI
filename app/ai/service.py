import app.ai.model as model
from app.ai.genetic import Genetic
import time
import datetime
import numpy as np
import torch as t
import threading

import plotly
import plotly.graph_objs as go

import json

class Service():
    inputs = None
    outputs = None
    date = None
    uid = None
    description = None
    online_learning = True
    batch_size=10
    lr=0.0001
    GAMMA=0.999
    opt='Adam'
    layers=[]
    active = True
    model = None
    genetic_learning = False
    mr=0.0001
    population_size=8
    genetic = None

    epoch=0
    batch=[]
    losses=[]

    def __init__(self,inputs=1,outputs=1,blanc=False):
        if blanc:
            pass
        self.date = str(datetime.datetime.now())

        self.inputs = inputs
        self.outputs = outputs

        self.model = model.Model_deep( self.inputs,
                                    self.outputs)
        self.update_genetic()

    def use_token(self,token):
        return self.genetic.use_token(token)

    def get_token(self):
        if self.free_tokens:
            token = self.free_tokens.pop(0) 
            return token
        else:
            return 'null'

    def copy(self):
        service = Service(self.inputs,self.outputs)
        service.layers = self.layers.copy()
        service.GAMMA = self.GAMMA
        service.batch_size= self.batch_size
        service.online_learning = self.online_learning
        service.date=self.date # may be real date
        service.description = 'tmp'
        service.lr = self.lr
        service.opt = self.opt
        service.active = self.active
        service.update_service()

        service.model = self.model.copy() # torch model must have copy() 
        return service

    def init_genetic(self):
        self.genetic = Genetic( self,
                                mr=self.mr,
                                population_size=self.population_size) 

    def update_genetic(self):
        if self.genetic_learning:
            self.init_genetic()
        else:
            self.genetic=None

    def update_service(self,form=None):
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
            self.model = model.Model_deep(self.inputs,self.outputs,
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
        
    def add(self,state,action,reward):
        if not self.online_learning:
            return None

        state = self.to_tensor(state)
        action = self.to_tensor(action)
        reward = self.to_tensor(reward)

        self.batch.append((state,action,reward))
        if len(self.batch) >self.batch_size:
            loss = self.train_on_batch()
            self.losses.append(loss)
            self.batch = []

    def train_on_batch(self):
        x = t.stack([t[0] for t in self.batch])
        y = t.stack([t[1] for t in self.batch])
        r = t.stack([t[2] for t in self.batch])

        loss = self.model.train(x,y,r)
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