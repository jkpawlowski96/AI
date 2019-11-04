import threading
import time
import numpy as np
import pandas as pd
from src.logg import dialog

class GeneticFit():
  lr=0.01
  population=4
  childrens=2
  epochs=10
  model=None
  model_create=None
  history=[]
  free_models=[]


  def get_model(self):
    return self.model
    # to use in threading
    if self.free_models:
      return self.free_models.pop()
    else:
      time.sleep(0.1)
      return self.get_model()
  
  def mutate(self,W,random=False,lr=None):
    if not lr:
      lr=self.lr
    if type(W)==list:
      for i in range(len(W)):
        W[i]=self.mutate(W[i],random)
      return W
    else:
      if random:
        return np.random.randn(*W.shape)
      return W + W*lr*np.random.randn(*W.shape)
   
  def cross(self,W1,W2):
    child = [(W1[i]+W2[i])/2 for i in range(len(W1))]
    return child
 
  def multiply(self,unit,childrens=None):
    if not childrens:
      childrens = self.childrens
    model = self.get_model()
    populate=[unit]
    W=unit['W']
    for _ in range(childrens):
      mW = self.mutate(W.copy())
      model.set_genom(mW)
      evaluate = model.evaluate()
      populate.append({'W':W,'reward':evaluate})
    self.free_models.append(model)
    self.populate += populate
  
  def fit(self): 

    self.lr_init = self.lr
    if not self.model:
      self.model = self.model_create()

    best_loss = 0
   
    stagnation_max = 10
    stagnation_counter = 0


    populate = [self.mutate(self.model.genom(), random = True) for c in range(self.population*self.population)]
    
    for epoch in range(self.epochs):
      
      self.lr = self.lr_init*(0.1**stagnation_counter)

      old_populate = populate.copy()
      populate = []
      for W in old_populate:
        self.model.set_genom(W)    
        evaluate = self.model.evaluate()
        populate.append({'W':W,'reward':evaluate})
        
      populate = sorted(populate, key = lambda i: i['reward'],reverse=True)
      populate =populate[:self.population]
      old_populate = populate.copy()
      
      self.populate = []
      tasks = []
      for unit in old_populate:
        #x = threading.Thread(target=self.multiply, args=(unit,childrens))
        #x.start()
        #tasks.append(x)
        self.multiply(unit)

      #while [t for t in tasks if t.is_alive()]:
      #  continue  

      populate=self.populate
      
      populate = sorted(populate, key = lambda i: i['reward'],reverse=True) 
      #print("populate:\\n",populate)

      n_best_loss = populate[0]['reward']
      #best_metrics = populate[0]['metrics']
      
      #if best_loss>=0.99:
      #  break
      
      if best_loss==n_best_loss:
        stagnation_counter += 1

        if stagnation_counter >= stagnation_max:
          break                      
      else:
        best_loss = n_best_loss
        stagnation_counter = 0
      
      populate = [unit['W'] for unit in populate[:self.population]]  # natural selection
      populate = [self.cross(populate[i],populate[i+1]) for i in range(len(populate)-1)] + [populate[0]]  # crossing species
      
      status ={'epoch':epoch,'reward':best_loss}
      dialog('Epoch['+str(epoch)+']'+' reward:'+str(best_loss))
      self.history.append(status)
  

    return self.model


