import json
import ast

def dict_to_json(data:dict):
    return json.dumps(data)

def json_to_dict(my_str:str):
    my_str=my_str.replace('"',"'")
    my_str=my_str.replace(': false',': False')
    my_str=my_str.replace(': null',': None')
    my_str = ast.literal_eval(my_str)
    my_dumps=json.dumps(my_str)
    return json.loads(my_dumps)

