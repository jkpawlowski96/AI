import threading
import time
import numpy as np
import pandas as pd

class GeneticFit():
  def __init__(self,create_model):
        self.lr = 0.1
        self.lr_init = self.lr
        self.history=[]
        self.jobs=1
        self.model=create_model()
        self.create_model= create_model
        self.free_models=[self.create_model() for x in range(self.jobs)]
  
  def get_model(self):
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
 
  def multiply(self,unit,childrens):
    model = self.get_model()
    populate=[unit]
    W=unit['W']
    for _ in range(childrens):
      mW = self.mutate(W.copy())
      model.set_weights(mW)
      evaluate = model.evaluate(self.X,self.y,verbose=False)
      populate.append({'W':mW,'loss':evaluate[0],'metrics':evaluate[1:]})
    self.free_models.append(model)
    self.populate += populate
  
  def fit(self,X,y): 
    model = self.model
    self.X=X
    self.y=y
    populate_size = 10
    childrens = 1
  
    epochs = 1000
    
    best_loss = 1
    best_metrics = 0
    stagnation_max = 10
    stagnation_counter = 0

    populate = [self.mutate(model.get_weights(),random = True) for c in range(populate_size)]
    
    for epoch in range(epochs):
      
      self.lr = self.lr_init*(0.1**stagnation_counter)

      old_populate = populate.copy()
      populate = []
      for W in old_populate:
        model.set_weights(W)    
        evaluate = model.evaluate(X,y,verbose=False)
        populate.append({'W':W,'loss':evaluate[0],'metrics':evaluate[1:]})
        
      populate = sorted(populate, key = lambda i: i['loss'],reverse=False)
      populate =populate[:populate_size]
      old_populate = populate.copy()
      
      self.populate = []
      tasks = []
      for unit in old_populate:
        #x = threading.Thread(target=self.multiply, args=(unit,childrens))
        #x.start()
        #tasks.append(x)
        self.multiply(unit,childrens)

      #while [t for t in tasks if t.is_alive()]:
      #  continue  

      populate=self.populate
      
      populate = sorted(populate, key = lambda i: i['loss'],reverse=False) 
      #print("populate:\\n",populate)

      n_best_loss = populate[0]['loss']
      best_metrics = populate[0]['metrics']
      
      if best_loss<=0.001:
        break
      
      if best_loss==n_best_loss:
        stagnation_counter += 1

        if stagnation_counter >= stagnation_max:
          break                      
      else:
        best_loss = n_best_loss
        stagnation_counter = 0
      
      populate = [unit['W'] for unit in populate[:populate_size]]  # natural selection
      populate = [self.cross(populate[i],populate[i+1]) for i in range(len(populate)-1)] + [populate[0]]  # crossing species
      
      status ={'epoch':epoch,'loss':best_loss,'metrics':best_metrics}
      print('Epoch[',epoch,']',' loss:',best_loss,' metrics:',best_metrics)
      self.history.append(status)

    model.set_weights(populate[0])
    self.history = pd.DataFrame(self.history)
    m = 0
    for metric in model.metrics_names[1:]:
      self.history[metric]=[x[m] for x in gf.history.metrics]
      m +=1
    return model


