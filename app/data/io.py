import pickle
from flask import make_response
import datetime

def load(service, form, data):
    if form == 'pickle':
        try:
            model = pickle.load(data)
            service.model = model
            service.layers = model.layers
            return True
        except:
            pass

    
    return False


def export(service, form):

    if form == 'pickle':
        data = pickle.dumps(service.model)

    resp = make_response(data)
    date = str(datetime.datetime.now()).replace(' ','_')
    resp.headers["Content-Disposition"] = f'attachment; filename={service.uid}{date}.{form}'
    resp.headers["Content-Type"] = "text/json"

    return resp