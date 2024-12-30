import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


application=Flask(__name__)
app=application
model=pickle.load(file=open(file='airfoil_model.pkl',mode='rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    data=np.array([[float(x) for x in request.form.values()]])
    output=model.predict(data)[0]
    return render_template('home.html',prediction_text=f"Airfoil pressure is {output}")


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(f'api-data:{data}')

    new_data=np.array([list(data.values())])
    output=model.predict(new_data)[0]
    return jsonify(output)

if __name__=='__main__':
    app.run(host="0.0.0.0")