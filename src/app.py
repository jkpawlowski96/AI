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


@app.route("/")
def home():
    """
    Homepage
    :return:
    """
    return '<h1>Hello World!</h1>'


if __name__ == "__main__":
    app.run(host="0.0.0.0")
