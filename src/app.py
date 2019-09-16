from flask import Flask, Response
from src.ai import AI
# flask app
app = Flask(__name__)

ai = AI()

@app.route("/")
def home():
	"""
	Homepage
	:return:
	"""
	return  '<h1>Hello World!</h1>'



if __name__ == "__main__":
	app.run(host="0.0.0.0")
	