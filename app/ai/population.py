import numpy as np

class Population():
    
    def __init__(self):
        self.pop = []


    def add(self,x):
        self.pop.append(x)

    def sort(self,key='Reward'):
        if key is 'Reward':
            self.sort_reward()

    def sort_reward(self):
        rewards = []
        for x in self.pop:
            rewards.append({'x':x,'r':x.reward_total})

        rewards = sorted(rewards, key = lambda i: i['r'],reverse=True) 

        self.pop = [i['x'] for i in rewards]
        
    def get(self,i=-1):
        if i >=0:
            if i in range(len(self.pop)):
                return self.pop[i]
        if len(self.pop)  is 0:
            return None
        if len(self.pop) > 1:
            i = np.random.randint(low=0,high=len(self.pop)-1)
        else:
            i = 0
        return self.pop[i]