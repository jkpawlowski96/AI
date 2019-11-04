import time
import datetime


class Model():
    inputs = None
    outputs = None
    date = None
    uid = None
    description = None

    def __init__(self):
        self.date = str(datetime.datetime.now())
