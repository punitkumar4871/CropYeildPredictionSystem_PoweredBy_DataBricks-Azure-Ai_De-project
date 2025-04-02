from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            data = CustomData(
                State_Name=request.form.get('State_Name'),
                District_Name=request.form.get('District_Name'),
                Season=request.form.get('Season'),
                Crop=request.form.get('Crop'),
                Crop_Year=int(request.form.get('Crop_Year')),
                Area=float(request.form.get('Area')),
                annual_rainfall=float(request.form.get('annual_rainfall')),
            )
            
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('home.html', results=str(results[0]))
        except Exception as e:
            return render_template('home.html', results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
