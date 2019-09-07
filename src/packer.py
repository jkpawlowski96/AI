import json

def dict_to_json(data:dict):
    return json.dumps(data)

def json_to_dict(data:json):
    return json.loads(data)

