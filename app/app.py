from app.data.database import Database
from flask import Flask, render_template, request, flash, redirect
import sys

app = Flask(__name__)

db = Database()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/hello")
def hello():
    return 'hello'


@app.route("/build", methods=['GET', 'POST'])
def build():
    if request.method == "POST":
        form = request.form
        resp = db.build_model(form)
        if resp is True:
            return index()
        else:
            flash(resp)
            return redirect(request.url)
    else:

        return render_template("build.html")


@app.route("/manage")
def manage():
    return render_template("manage.html", data=db)


@app.route("/<string:uid>", methods=['GET', 'POST'])
def model(uid):
    if uid not in db.uids:
        return redirect("/")
    model = db.models[uid]
    if request.method == 'GET':
        return render_template("model.html", model=model)

    else:  # POST
        form = request.form

        model.options(form.getlist('options'))

        return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
