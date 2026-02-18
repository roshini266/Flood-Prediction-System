from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("floods.save")
sc = joblib.load("transform.save")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        cloud = float(request.form["cloud"])
        annual = float(request.form["annual"])
        janfeb = float(request.form["janfeb"])
        marchmay = float(request.form["marchmay"])
        junsep = float(request.form["junsep"])

        data = np.array([[cloud, annual, janfeb, marchmay, junsep]])

        data = sc.transform(data)

        result = model.predict(data)[0]

        if result == 1:
            return render_template("chance.html")
        else:
            return render_template("nochance.html")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

