from flask import Flask, render_template, request, flash, redirect
from data.database import Database

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

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


@app.route("/manage", methods=['GET', 'POST'])
def manage():

    return render_template("manage.html", data=db)


if __name__ == "__main__":
    app.run(debug=True)
