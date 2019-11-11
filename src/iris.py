import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from genetic import GeneticFit
enc = preprocessing.LabelEncoder()

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
df.columns = ['a','b','c','d','type']
df.type = enc.fit_transform(df.type)

X = df[['a','b','c','d']].values
y = df.type.values.reshape(-1,1)
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)


class Model:
    def __init__(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(10,input_dim=4,activation='linear'))
        model.add(tf.keras.layers.Dense(10,activation='relu'))
        model.add(tf.keras.layers.Dense(3,activation='softmax'))
            
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])
        self.model = model

    def evaluate(self):
        return self.model.evaluate(X_train,y_train,verbose=False)   

    def genom(self):
        return self.model.get_weights()

    def set_genom(self,genom):
        self.model.set_weights(genom)

    metrics=['acc']

    




gf = GeneticFit()

gf.model_create=Model
gf.lr=0.01
gf.population=10
gf.childrens=2


model = gf.fit().model

model.evaluate(X_train,y_train)
model.evaluate(X_test,y_test)