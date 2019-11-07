import numpy as np
from app.ai.service import Service


class Database():
    services = {}
    uids = []

    def __init__(self):
        self.add_service('ai_2',7,5,'Manipulator simulated by Unity 3D Engine')

    def build_service(self, form: dict):
        uid = form['uid']
        if uid in self.uids:
            return 'Uid is already in use'
        try:
            inputs = np.int(form['inputs'])
            outputs = np.int(form['outputs'])
        except:
            return 'Empty numbers!'

        for x in [inputs, outputs]:
            if x < 1:
                return 'Numbers must be > 0'

        description = form['description']

        self.add_service(uid, inputs, outputs, description)
        return True

    def add_service(self, uid, inputs, outputs, description):
        if uid in self.uids:
            return False
        self.uids.append(uid)

        service = Service(inputs,outputs)
        service.uid = uid
        service.description = description

        self.services[uid] = service
