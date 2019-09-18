import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import json
from config import Config 
from logg import dialog
from operator import itemgetter

class AI:
    def __init__(self,config:json=None):
        

        self.id = 0
        self.config = Config(config)

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(units=4, input_dim = 3, activation='relu' ))
        self.model.add(keras.layers.Dense(units=4, activation='relu' ))
        self.model.add(keras.layers.Dense(units=1, activation='linear' ))
        opt = keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(loss='mean_squared_error', optimizer=opt)


    def config(self, config:json=None):
        self.config = Config(config)
       

    def forward(self, data:dict):
        X = data['X']
        X = np.array([X],dtype=np.float32)
        pred = self.model.predict(X)[0][0]
        dialog(pred)
        return pred

        
    def train(self, batch):
        batch = sorted(batch, key = lambda i: i['Reward'],reverse=True) 

        batch = batch[:5]

        State = [frame['State'] for frame in batch]
        Q = [frame['Q'] for frame in batch]
        #Reward = [frame['Reward'] for frame in batch]

        X=[]
        Y=[]

        # X amd Y selection
        X = State
        Y = Q


        for x,y in zip(X,Y):
            x = np.array([x],dtype=np.float32)
            y = np.array([y],dtype=np.float32)
            #dialog(x)
            #dialog(y)
            self.model.train_on_batch(x,y)




