import numpy as np
from flask import Flask, request, jsonify, render_template

import joblib
app = Flask(__name__)
model = joblib.load("iris.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = [features]
    output = model.predict(features)
    if output==0:
        output='Setosa'
    elif output==1:
        output='Versicolor'
    else :
        output ="Virginica"

    return render_template('index.html', prediction_text='Predicted type of the plant is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run()