import torch as t 

from app.ai.population import Population

class Genetic():
    population_size = 8
    mr = 0.001

    service = None
    pop = Population()

    def __init__(self, *args, **kwargs):
        for arg in args:
            break
        for arg in kwargs:
            if arg == 'service':
                self.service=kwargs[arg]
            if arg == 'population_size':
                self.population_size=kwargs[arg]
            if arg == 'mr':
                self.mr=kwargs[arg]

        self.pop = Population(self.population_size)
        
    
    
    def new_population(self):
        _pop = Population() # new empty population
        self.pop.sort() # by reward as default

        for i in range(self.population_size):
            x = self.cross(self.pop.get(i),self.pop.get()) # new child
            _pop.add(x) # add child to new populate
        

    def cross(self,x,y):
        child = x.copy()
        child.model.cross(y.model)
        return child
