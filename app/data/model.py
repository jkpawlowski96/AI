import time
import datetime


class Model():
    inputs = None
    outputs = None
    date = None
    uid = None
    description = None
    online_learning = None
    active = True

    def __init__(self):
        self.date = str(datetime.datetime.now())

    def options(self, options):
        if 'online_learning' in options:
            self.online_learning = True
        else:
            self.online_learning = False
