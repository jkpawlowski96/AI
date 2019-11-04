from flask import Flask, Response
import src.ai as ai
import src.packer as packer
from src.logg import dialog
from src.formats import Callback
import threading
import numpy as np
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


tasks=[]

# flask app

# env\Scripts\activate

app = Flask(__name__)

AI = ai.AI()

@app.route("/")
def home():
    """
    Homepage
    :return:
    """
    return '<h1>Hello World!</h1>'


	#dialog(batch)
	AI.train(batch)

	return 'done'

@app.route("/config/<string:config>")
def config():
	#dialog(ai.driver.to_evaluate)
	try:
		uid = ai.driver.to_evaluate.pop(0)
		ai.driver.evaluating.append(uid)
		return str(uid)
	except:
		return 'none'


@app.route("/<string:uid>/callback/<string:reward>")
def callback(uid,reward):
	reward = reward.replace(',','.')
	reward = np.float(reward)

	#dialog(uid+' -> reward '+str(reward))
	
	ai.driver.simulations[uid]={'Reward':reward}
	ai.driver.evaluating.remove(uid)

	return 'done'


@app.route("/<string:uid>/forward/<path:data>")
def forward(uid,data):
	data= str(data)
	#dialog(data)
	data = packer.json_to_dict(data)
	pred = AI.forward(uid,data)
	return str(pred)

if __name__ == "__main__":
	app.run(host="0.0.0.0")


	