import pandas as pd
from flask import Flask, request
from sklearn.externals import joblib

with open('mdl.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    Bedroom = request.form['Bedroom']
    Baths = request.form['Baths']
    TotalRooms = request.form['TotalRooms']
    Location = request.form['Location']
    LandSize = request.form['LandSize']
    LivingArea = request.form['LivingArea']

    prediction = model.predict(pd.DataFrame([[Bedroom, Baths, TotalRooms, Location, LandSize, LivingArea]]))
    return str(prediction)

if __name__=='__main__':
    app.run(port=5001, debug=True)