import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import json
from src.config import Config 
import time
from src.logg import dialog
from operator import itemgetter
import src.formats as formats
from src.genetic import GeneticFit
import src.driver as driver

import threading

one_driver=driver.Driver()
app_models={}

tasks=[]

class Model:
    def __init__(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(10,input_dim=3,activation='relu'))
        model.add(tf.keras.layers.Dense(10,activation='relu'))
        #model.add(tf.keras.layers.Dense(5,activation='relu'))
        model.add(tf.keras.layers.Dense(1,activation='tanh'))
            
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])
        self.model = model

    def evaluate(self):
        #return self.model.evaluate(X_train,y_train,verbose=False) 
        uid = np.random.randint(0,99999999)  # random unit id of simulation
        app_models[str(uid)]=self  # assign model to global app models 
        
        # EVALUATING
        reward = one_driver.evaluate(uid)

        app_models[str(uid)]=None  # delete model from app models
        dialog(reward)
        return reward

    def genom(self):
        return self.model.get_weights()

    def set_genom(self,genom):
        self.model.set_weights(genom)

    def forward(self,X):
        return self.model.predict(X)


    metrics=['acc']

class AI:
    def __init__(self,config:json=None):
        
        self.model = Model() 

        x = threading.Thread(target=self.genetic_fit)
        x.start()
        tasks.append(x)

    def config(self, config:json=None):
        self.config = Config(config)
       

    def forward(self,uid, data:dict):
        model = app_models[str(uid)]

        X = data['X']
        X = np.array([X],dtype=np.float32)
        pred = model.forward(X)[0][0]
        #dialog(pred)
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

    def genetic_fit(self):
        self.gen_fit=GeneticFit()

        def return_model():
            return self.model
        self.gen_fit.model_create=return_model  # create driver to unity and evaluate in the future

        self.gen_fit.lr=0.05
        self.gen_fit.population=10
        self.gen_fit.childrens=3
        self.gen_fit.epochs=500


        self.gen_fit.fit()
        dialog('GENETIC FIT COMPLETED')
        self.model = self.gen_fit.model


        

