from app.data.database import Database
from flask import Flask, render_template, request, flash, redirect
import sys

app = Flask(__name__)

db = Database()


@app.route("/")
def index():
    return render_template("index.html")



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

        
        model.update_model(form)
        return redirect(request.url)


@app.route("/use/<string:uid>/<string:data>")
def model_use(uid,data):
    if uid not in db.uids:
        return 'null'
    data = data.replace(",",".")
    if ';' in data:
        x = data.split(";")[0].split()
        state = data.split(";")[1].split()
        reward = data.split(";")[2].split()
        db.models[uid].add(state,reward)
    else:
        x = data.split()
    return db.models[uid].forward(x)
     
@app.route("/layer/<string:uid>/<string:option>/<int:layer>")
def model_layer(uid,option,layer):
    if option=='del':
        db.models[uid].layers.pop(layer)
        db.models[uid].update_model()

    if option=='add':
        db.models[uid].layers.insert(layer+1,1)
        db.models[uid].update_model()

    return redirect("/"+uid)

if __name__ == "__main__":
    app.run(debug=True)
