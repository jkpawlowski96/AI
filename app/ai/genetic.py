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
        self.history = {'epoch':[],
                        'reward_total':[]
                        }

        self.service = service
        self.population_size = lambda: self.service.population_size
        self.mr = lambda: self.service.mr

        self.init_population()
       
        
    def init_population(self):
        self.pop = Population()
        for _ in range(self.population_size()):
            x = self.service.copy()
            x = self.mutate(x,mr=10)
            self.pop.add(x)

        self.init_tokens()
    
    def init_tokens(self):
        self.tokens = []
        self.tokens_use = []
        self.tokens_free = []
        for _ in range(len(self.pop.pop)):
            token = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            self.tokens.append(token)
        self.tokens_free = self.tokens.copy()
        
    def finish(self,token,reward):
        if token in self.tokens_use:
            self.use_token(token).reward_total = reward
            self.tokens_use.remove(token)
        if not self.tokens_use and not self.tokens_free:
            self.evolve_population()

    def use_token(self,token):
        index = self.tokens.index(token)
        x = self.pop.get(index)
        return x

    def free_token(self):
        if self.tokens_free:
            token =  self.tokens_free.pop()
            self.tokens_use.append(token)
            return token
        else:
            return 'null'

    def mutate(self, x, mr = None):
        _x = x.copy()
        if not mr:
            mr = self.mr()
        state = x.model.state_dict()
        for k in state.keys():
            state[k] = state[k] + (t.rand_like(state[k])*2-1)*mr

        _x.model.load_state_dict(state)
        return _x

    def cross(self,x,y):
        child = x.copy()
        state = x.model.state_dict()
        _state = y.model.state_dict()
        for k in state.keys():
            x = state[k]
            y = _state[k]
            state[k] = (x+y)/2
            
        child.model.load_state_dict(state)

        return child

    def evolve_population(self):
        pop = self.pop
        #self.init_population() # new empty population
        self.pop = Population()
        pop.sort() # by reward as default
        best = pop.get(0)
        self.service.model = best.model.copy()
        
        self.pop.add(best)
        self.history['reward_total'].append(best.reward_total)
        childrens = 2
        survived = self.population_size()/childrens
        survived = np.int(survived)
        for i in range(survived):
            x = self.cross(pop.get(i),pop.get()) # new child

            x = self.mutate(x)
            self.pop.add(x) # add child to new populate

            x = self.mutate(x)
            self.pop.add(x) # add child to new populate

        self.init_tokens()


    def plot_reward_total(self):
        return plot.linear(self.history['reward_total'])
