import torch as t
import random
import string
from app.ai.population import Population
import numpy as np
import app.ai.plot as plot


class Genetic():

    def __init__(self, service):
        self.tokens = []
        self.tokens_free = []
        self.tokens_use = []
        self.service = None
        self.best = None
        self.cross_method = 'dna'
        self.history = {'epoch': [],
                        'reward_total': [],
                        'batch_loss':[]
                        }

        self.service = service
        self.population_size = lambda: self.service.population_size
        self.childrens = 4
        self.mr = lambda: self.service.mr
        self.psi = 0.01
        self.init_population()

    def init_population(self):
        self.pop = Population()
        self.pop.add(self.service.copy())
        for _ in range(self.population_size()-1):
            x = self.service.copy()
            x = self.mutate_randn(x, random=True)
            #x = self.mutate(x, mr=0.5)
            self.pop.add(x)
        self.best = self.pop.get()
        self.init_tokens()

    def init_tokens(self):
        self.tokens = []
        self.tokens_use = []
        self.tokens_free = []
        for _ in range(len(self.pop.pop)):
            token = ''.join(random.choice(
                string.ascii_uppercase + string.digits) for _ in range(10))
            self.tokens.append(token)
        self.tokens_free = self.tokens.copy()

    def finish(self, token, reward):
        if token in self.tokens_use:
            self.use_token(token).reward_total = reward
            self.tokens_use.remove(token)
        if not self.tokens_use and not self.tokens_free:
            self.evolve_population()

    def use_token(self, token):
        index = self.tokens.index(token)
        x = self.pop.get(index)
        return x

    def free_token(self):
        if self.tokens_free:
            token = self.tokens_free.pop()
            self.tokens_use.append(token)
            return token
        else:
            return 'null'

    def mutate_randn(self, x, mr=None,random=False):
        _x = x.copy()
        if not mr:
            mr = self.mr()
        state = x.model.state_dict()
        for k in state.keys():
            state[k] = state[k] + state[k]*(t.rand_like(state[k])*2-1)*mr
            if random:
                state[k] = (t.rand_like(state[k])*2-1)

        _x.model.load_state_dict(state)
        return _x

    def mutate(self, x, mr=None):
        _x = x.copy()
        if not mr:
            mr = self.mr()
        state = x.model.state_dict()

        i = np.random.randint(low=0,high=len(state))
        k = list(state.keys())[i]

        select = t.rand_like(state[k]) <= mr
        filtr = t.zeros_like(state[k])
        filtr[select] = 1
        
        mut = (t.rand_like(state[k])*2)-1
        mut = mut * filtr

        state[k] = state[k] + mut * self.psi
            
        _x.model.load_state_dict(state)
        return _x

    def cross(self, x, y):
        if self.cross_method is 'dna':
            return self.cross_dna(x, y)
        if self.cross_method is 'mean':
            return self.cross_mean(x, y)

    def cross_mean(self, x, y):
        child = x.copy()
        state = x.model.state_dict()
        _state = y.model.state_dict()
        for k in state.keys():
            x = state[k]
            y = _state[k]
            state[k] = (x+y)/2

        child.model.load_state_dict(state)

        return child

    def cross_dna(self, x, y):
        child = x.copy()
        state = x.model.state_dict()
        _state = y.model.state_dict()
        for k in state.keys():
            x = state[k]
            y = _state[k]

            choice = t.randint_like(x, 0, 2)
            choice_n = (choice - 1) * -1

            state[k] = x * choice + y * choice_n

        child.model.load_state_dict(state)

        return child

    def train_on_baches(self, pop):
        service = self.best.copy()
        model = service.model
        model.optimizer.zero_grad()
        loss = 0
        for s in pop:
            # load weights form populated model
            w = s.model.state_dict()
            model.load_state_dict(w)
            # train model on batch and acumulate loss
            x,y,r = s.data_from_batch()
            loss += model.loss(x,y,r)
        loss.backward()
        model.optimizer.step()
        service.model = model
        self.history['batch_loss'].append(loss.item())
        return service

    def evolve_population(self):
        pop = self.pop
        # self.init_population() # new empty population
        self.pop = Population()
        pop.sort()  # by reward as default
        if self.best.reward_total < pop.get(0).reward_total:
            self.best = pop.get(0).copy()
            self.best.reward_total = pop.get(0).reward_total
            self.service.model = self.best.model.copy()
        
        self.history['reward_total'].append(self.best.reward_total)
        childrens = self.childrens
        survived = (self.population_size()-2)/childrens
        survived = np.int(survived)
        for i in range(survived):
            x = pop.get(i)
            _x = pop.get(i+1)
            #x = self.cross_mean(x,_x)
            #x = self.cross_dna(x,_x)  # new dna
            for _ in range(childrens):
                child = self.cross_dna(x,_x)  # new dna
                child = self.mutate(child)
                self.pop.add(child)  # add child to new populate

        self.pop.add(self.best.copy())  # best model
        
        if self.service.online_learning:
            x = self.train_on_baches(pop.pop)
            self.pop.add(x)  # best model trained on batches
        else:
            x = self.mutate(self.best)
            self.pop.add(x)
        self.init_tokens()

    def plot_reward_total(self):
        return plot.linear(self.history['reward_total'])

    def plot_batch_loss(self):
        return plot.linear(self.history['batch_loss'])
