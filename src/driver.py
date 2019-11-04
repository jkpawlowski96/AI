import time
import numpy as np

simulations={}
evaluating=[]
to_evaluate=[]

class Driver:
    def __init__(self,uid=None):
        
        self.reset()
        self.config(uid)

    def config(self,uid):
        return NotImplemented

    def reset(self):
        return NotImplemented

    def evaluate(self,uid):
        uid = str(uid)
        to_evaluate.append(uid)
        
        while uid in to_evaluate:
            time.sleep(1)

        while uid in evaluating:
            time.sleep(1)

        reward = simulations[uid]['Reward']

        reward = np.float(reward)

        return reward


        
        
