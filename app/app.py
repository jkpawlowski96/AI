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
        resp = db.build_service(form)
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
    service = db.services[uid]
    if request.method == 'GET':
        return render_template("model.html", model=service)

    else:  # POST
        form = request.form

        
        service.update_service(form)
        return redirect(request.url)


@app.route("/use/<string:uid>/<string:data>")
def model_use(uid,data):
    if uid not in db.uids:
        return 'null'
    data = data.replace(",",".")
    if ';' in data:
        x = data.split(";")[0].split()
        state = data.split(";")[1].split()
        action = data.split(";")[2].split()
        reward = data.split(";")[3].split()
        db.services[uid].add(state,action,reward)
    else:
        x = data.split()
    return db.services[uid].forward(x)
     
@app.route("/layer/<string:uid>/<string:option>/<int:layer>")
def model_layer(uid,option,layer):
    if option=='del':
        db.services[uid].layers.pop(layer)
        db.services[uid].update_services()

    if option=='add0':
        option='add'
        layer=-1

    if option=='add':
        db.services[uid].layers.insert(layer+1,1)
        db.services[uid].update_services()



    return redirect("/"+uid)

@app.route("/history/<string:uid>/<string:option>")
def model_history(uid,option):
    if option=='clear':
        db.services[uid].losses=[]
        db.services[uid].epoch=0
    return redirect("/"+uid)

if __name__ == "__main__":
    app.run(debug=True)
