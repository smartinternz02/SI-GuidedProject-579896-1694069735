import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details') # rendering the html template
def index() :
    return render_template('details.html')

@app.route('/Crop_predict', methods=['GET','POST'])
def predict() :
    # loading model which we saved
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature= float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph= float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    


    prediction =model.predict(pd.DataFrame([[N,P,K, temperature, humidity, ph, rainfall]], columns= ['N', 'P', 'K', 'temperature', 'humidity', 'ph',
       'rainfall']))
    
    
    
    
    return render_template('crop_predict.html', prediction_text ="{}".format(prediction))



if __name__ == '__main__':
    app.run()