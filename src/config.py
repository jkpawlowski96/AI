from src.packer import json_to_dict, dict_to_json
import json
import numpy as np

class Config:
    def __init__(self,config:json):
        if not config:
            #  default json configuration
            config = '{"Inputs": ["x1", "x2"], "Outputs": ["y"]}'
        
        config = json_to_dict(config)  # json to dictionary
        
        self.inputs_labales = config['Inputs']
        self.outputs_labels = config['Outputs']
        self.inputs_shape = np.shape(self.inputs_labales)
        self.outputs_shape = np.shape(self.outputs_labels)

