import numpy as np
#import keras
import json
from config import Config 

class AI:
    def __init__(self,config:json=None):
        self.id = 0
        self.config = Config(config)

    def config(self, config:json=None):
        self.config = Config(config)

    def forward(self, state:json):
        return 1
        
ai = AI()
ai.forward(None)

