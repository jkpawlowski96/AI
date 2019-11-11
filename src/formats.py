class Callback:
    def __init__(self,data:dict=None):
        if data:
            self.unit = data['Unit']
            self.reward = data['Reward']
        

              