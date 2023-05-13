from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
from scipy.special import inv_boxcox

data_dict=None
df=pd.DataFrame(pickle.load(open("dataframe.pkl",'rb')))
pipe=pickle.load(open("pipe.pkl",'rb'))

app=Flask(__name__)

@app.route('/')
def index():
    global data_dict
    data_dict={
        'model_name':df['model_name'].unique(),
        'model_year':pd.Series(df['model_year'].unique()).sort_values(),
        'kms_driven':pd.Series(df['kms_driven'].unique()).sort_values(),
        'owner':df['owner'].unique(),
        'location':df['location'].unique(),
        'mileage':df['mileage'].unique(),
        'power':df['power'].unique(),
        'prediction':False
    }
    return render_template("index.html",data_dict=data_dict)

@app.route('/predict',methods=['POST'])
def predict():
    global data_dict
    model_name=request.form['model_name']
    model_year=request.form['model_year']
    kms_driven=request.form['kms_driven']
    owner=request.form['owner']
    location=request.form['location']
    mileage=request.form['mileage']
    power=request.form['power']

    interpreted_values={
        'model_name':model_name,
        'model_year':model_year,
        'kms_driven':kms_driven,
        'owner':owner,
        'location':location,
        'mileage':mileage,
        'power':power
    }


    prediction=pipe.predict(np.array([model_name,model_year,kms_driven,owner,location,mileage,power],dtype=object).reshape(1,7))

    # inversing the sqaure root transformation
    prediction=(prediction**2)[0]

    data_dict['prediction']=prediction

    return render_template('index.html',data_dict=data_dict)


if __name__=='__main__':
    app.run(debug=True)