import app.ai.model as model
from app.ai.genetic import Genetic
import app.ai.plot as plot
import time
import datetime
import numpy as np
import torch as t
import threading
import sys


class Service():

    def __init__(self, inputs=1, outputs=1, main_service=False):
        self.main_service = main_service
        self.uid = None
        self.description = None
        self.online_learning = True
        self.batch_size = 10
        self.lr = 0.0001
        self.GAMMA = 0.999
        self.opt = 'SGD'
        self.layers = []
        self.active = True

        self.genetic_learning = False
        self.mr = 0.1
        self.population_size = 4
        self.genetic = None

        self.reward_total = 0

        self.epoch = 0
        self.batch = []
        self.losses = []

        self.date = str(datetime.datetime.now())

        self.inputs = inputs
        self.outputs = outputs

        self.model = model.Model_deep(self.inputs,
                                      self.outputs)
        self.update_genetic()

    def use_token(self, token):
        return self.genetic.use_token(token)

    def get_token(self):
        return self.genetic.free_token()

    def plot_losses(self):
        return plot.linear(self.losses)

    def copy(self):
        service = Service(self.inputs, self.outputs)
        service.layers = self.layers.copy()
        service.GAMMA = self.GAMMA
        service.batch_size = self.batch_size
        service.online_learning = self.online_learning
        service.date = self.date  # may be real date
        service.description = 'tmp'
        service.lr = self.lr
        service.opt = self.opt
        service.active = self.active
        service.update_service()

        service.model = self.model.copy()  # torch model must have copy()
        return service

    def init_genetic(self):
        self.genetic = Genetic(service=self)

    def update_genetic(self):
        if self.genetic_learning and not self.genetic:  # start genetic
            self.init_genetic()
        if not self.genetic_learning:  # remove gentic
            self.genetic = None

    def update_service(self, form=None):
        self.update_genetic()
        if form is not None:
            # checklist
            self.options(form.getlist('options'))

            form = form.to_dict()
            # q-learning
            if self.online_learning:
                try:
                    self.lr_percent = form['lr_percent']
                
                    self.lr = np.float(form['lr'])
                
                    self.opt = form['opt']
    
                    self.GAMMA = np.float(form['GAMMA'])
                
                    self.batch_size = np.int(form['batch_size'])
                except:
                    pass
            # genetic
            if self.genetic_learning:
                if 'mr' in form.keys():
                    self.mr = np.float(form['mr'])
                if 'psi' in form.keys():
                    self.genetic.psi = np.float(form['psi'])
                if 'childrens' in form.keys():
                    self.genetic.childrens = np.int(form['childrens'])   
                if 'population_size' in form.keys():
                    self.population_size = np.int(form['population_size'])

            # nn configuration
            for n in range(len(self.layers)):
                try:
                    l = form['l'+str(n)]
                    l = np.int(l)
                    if l <= 0:
                        self.layers = self.layers[:n-1]
                        pass
                    self.layers[n] = l
                except Exception:
                    self.layers = self.layers[:n-1]
                    pass

        if self.layers is not self.model.layers:
            self.model = model.Model_deep(self.inputs, self.outputs,
                                          layers=self.layers.copy())

        if self.lr is not self.model.lr:
            self.model.update_optimizer(lr=self.lr)

        if self.GAMMA is not self.model.GAMMA:
            self.model.GAMMA = self.GAMMA

        if self.opt is not self.model.opt:
            self.model.update_optimizer(opt=self.opt)

    def options(self, options):
        if 'online_learning' in options:
            self.online_learning = True
        else:
            self.online_learning = False

        if 'genetic_learning' in options:
            self.genetic_learning = True
        else:
            self.genetic_learning = False
        self.update_genetic()

    def finish(self, token, data):
        if not self.genetic:
            return 'null'

        data = data.split('$')[1]
        data = data.replace(',', '.')
        reward = np.float(data)
        self.genetic.finish(token, reward)

    def forward(self, x):
        x = self.to_tensor(x)
        x = self.model.forward(x.view((1, -1)))
        return self.from_tensor(x)

    def add(self, state, action, reward):
        if not self.online_learning:
            return None
        if self.main_service:
            return None

        state = self.to_tensor(state)
        action = self.to_tensor(action)
        reward = self.to_tensor(reward)

        self.batch.append((state, action, reward))
        # if len(self.batch) > self.batch_size:
        #    loss = self.train_on_batch()
        #    self.losses.append(loss)
        #    self.batch = []

    def train_on_batch(self):
        x, y, r = data_from_batch()
        loss = self.model.train(x, y, r)
        #loss = self.model.train_loss(x, y, r)
        self.batch = []
        return loss

    def data_from_batch(self):
        x = t.stack([t[0] for t in self.batch])
        y = t.stack([t[1] for t in self.batch])
        r = t.stack([t[2] for t in self.batch])
        return x, y, r

    def to_tensor(self, x):
        x = np.array(x).astype(np.float)

        #x = [np.float(v) for v in x]
        x = t.FloatTensor(x)
        return x

    def from_tensor(self, x):
        #x = x.round()
        resp = ""
        for v in x.view(-1):
            resp += str(v.item())+";"
        resp = resp[:-1]
        resp = resp.replace(".", ",")
        return resp

    def n_layers(self):
        return len(self.layers)
