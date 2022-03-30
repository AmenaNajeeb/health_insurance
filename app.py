#!/usr/bin/env python
# coding: utf-8

import flask
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
@app.route('/index')

def index():
    return flask.render_template('index.html')

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 6)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result)== 1:
            prediction ='Genuine Application'
        else:
            prediction ='Fraudulent Application'
        
        return render_template("result.html", prediction = prediction)

if __name__ == '__main__':
	app.run(debug=True)