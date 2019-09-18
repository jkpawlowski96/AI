from flask import Flask, Response
from ai import AI
import packer
from logg import dialog

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# flask app

# env\Scripts\activate

app = Flask(__name__)

ai = AI()

@app.route("/")
def home():
	"""
	Homepage
	:return:
	"""
	return  '<h1>Hello World!</h1>'

@app.route("/add/<path:data>")
def add(data):
	data_list = str(data).split('*')
	#dialog(data_list[0])
	batch = []
	for data in data_list:
		data = packer.json_to_dict(data)
		batch.append(data)
		#dialog(data)

	#dialog(batch)
	ai.train(batch)

	return 'done'

@app.route("/forward/<path:data>")
def forward(data):
	data= str(data)
   
	dialog(data)
	data_dict = packer.json_to_dict(data)
	#dialog(data)
	pred = ai.forward(data_dict)
	#dialog(pred)
	return str(pred)

if __name__ == "__main__":
	app.run(host="0.0.0.0")
	