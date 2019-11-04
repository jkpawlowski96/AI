import numpy as np
from app.data.model import Model


class Database():
    models = {}
    uids = []

    def build_model(self, form: dict):
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

        self.add_model(uid, inputs, outputs, description)
        return True

    def add_model(self, uid, inputs, outputs, description):
        if uid in self.uids:
            return False
        self.uids.append(uid)

        model = Model()
        model.uid = uid
        model.inputs = inputs
        model.outputs = outputs
        model.description = description

        self.models[uid] = model
