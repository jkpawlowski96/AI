from app.data.database import Database
from flask import Flask, render_template, request, flash, redirect
import sys
import app.data.io as io
from flask_dropzone import Dropzone

app = Flask(__name__)
dropzone = Dropzone(app)

db = Database()
db.add_service('ai_1', 3, 3, 'Self driven car simulated by Unity 3D Engine')
db.add_service('ai_2', 6, 4, 'Manipulator simulated by Unity 3D Engine')



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/export/<string:form>/<string:uid>")
def export(form,uid):
    if uid not in db.uids:
        return redirect("/")
    
    return io.export(db.services[uid],form)

@app.route("/import/<string:form>/<string:uid>", methods=['GET', 'POST'])
def dropzone(form,uid):
    if form == 'form':
        form = dropzone.form
        uid = dropzone.uid
        
    service = db.services[uid]
    service.form = form

    if request.method == "POST":

        f = request.files['file']
        #f.save(os.path.join('the/path/to/save', f.filename))
        if io.load(service, form, f):
            return redirect("/"+uid)
        else:
            return redirect("/")
    else:
        dropzone.uid = uid
        dropzone.form = form
        return render_template("load.html", s = service)

@app.route("/load/<string:form>/<string:uid>")
def load(form,uid):
    if uid not in db.uids:
        return redirect("/")
    
    service = db.services[uid]
    data = 0
    if io.load(service, form, data):
        return redirect("/"+uid)
    else:
        return redirect("/")

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
def service(uid):
    if uid not in db.uids:
        return redirect("/")
    service = db.services[uid]
    if request.method == 'GET':
        return render_template("service.html", s=service)

    else:  # POST
        form = request.form
        service.update_service(form)
        return redirect(request.url)


@app.route("/use/<string:uid>/<string:data>")
def service_use(uid, data):
    if uid not in db.uids:
        return 'null'
    return service_work(data, db.services[uid])


@app.route("/token/<string:uid>")
def get_token(uid):
    if uid not in db.uids:
        return 'null'
    service = db.services[uid]

    return service.get_token()


@app.route("/use/<string:uid>/<string:token>/<string:data>")
def service_use_token(uid, token, data):
    if uid not in db.uids:
        return 'null'
    service = db.services[uid]

    if '$' in data:  # experiment finished
        service.finish(token, data)
        return 'null'
    else:
        service = service.use_token(token)
        return service_work(data, service)


def service_work(data, service):
    data = data.replace(",", ".")
    if '*' in data:
        x = data.split("*")[0].split(';')
        state = data.split("*")[1].split(';')
        action = data.split("*")[2].split(';')
        reward = data.split("*")[3].split(';')
        service.add(state, action, reward)
    else:
        x = data.split(';')
    return service.forward(x)


@app.route("/layer/<string:uid>/<string:option>/<int:layer>")
def service_layer(uid, option, layer):
    if option == 'del':
        db.services[uid].layers.pop(layer)
        db.services[uid].update_service()

    if option == 'add0':
        option = 'add'
        layer = -1

    if option == 'add':
        db.services[uid].layers.insert(layer+1, 1)
        db.services[uid].update_service()

    return redirect("/"+uid)


@app.route("/history/<string:uid>/<string:option>")
def service_history(uid, option):
    if option == 'batch_loss':
        db.services[uid].genetic.history['batch_loss'] = []
    if option == 'reward_total':
        db.services[uid].genetic.history['reward_total'] = []
    return redirect("/"+uid)


@app.route("/restart_genetic/<string:uid>")
def service_genetic_restart(uid):
    service = db.services[uid]
    service.init_genetic()
    return redirect("/"+uid)

if __name__ == "__main__":
    app.run(debug=True,host='127.0.0.1')
