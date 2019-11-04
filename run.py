from app import app
from flask import Flask

if __name__ == "__main__":
    app = Flask(__name__)
    app.run(host="0.0.0.0")
