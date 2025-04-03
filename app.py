from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import google.generativeai as genai
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import time
from datetime import datetime
import textwrap

# Configure Gemini AI API Key
genai.configure(api_key="AIzaSyCzEHr9A5O7I-rZ8hAosbnAaKx0MUZ40k0")

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

def generate_ai_content(state, district, season, crop, area, predicted_yield):
    """Generate comprehensive farming recommendations using Gemini AI with delays between requests."""
    total_production = round(predicted_yield * area, 2)

    print(f"\nYield Information:")
    print(f"Predicted Yield: {predicted_yield} tons/hectare")
    print(f"Total Production: {area} hectares × {predicted_yield} = {total_production} tons\n")

    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompts = {
        'soil': f"Provide detailed soil recommendations for {crop} in {district}, {state} during {season}. Limit response to 150 words.",
        'pest': f"List common pests and diseases affecting {crop} in {district}. Limit response to 150 words.",
        'irrigation': f"Detail optimal irrigation practices for {crop} in {season}. Limit response to 150 words.",
        'practices': f"Describe best farming practices for {crop} in {state} during {season}. Limit response to 150 words.",
        'climate': f"Explain climate impacts on {crop} production in {district} for {season}. Limit response to 150 words.",
        'market': f"Analyze current market trends for {crop} in {state}. Limit response to 150 words.",
        'schemes': f"List government schemes for {crop} farmers in {district}, {state}. Limit response to 150 words."
    }

    responses = {}
    for key, prompt in prompts.items():
        for attempt in range(3):  # Retry up to 3 times if quota is exceeded
            try:
                time.sleep(2)  # ⏳ Delay of 2 seconds between each AI request
                response = model.generate_content(prompt)
                responses[key] = response.text if response.text else "No data available."
                break  # If successful, break the retry loop
            except Exception as e:
                error_msg = str(e).lower()
                if "quota exceeded" in error_msg or "429" in error_msg:
                    responses[key] = "⚠️ AI-generated content is currently unavailable due to high demand. Please try again later."
                    break  # Stop retrying and set a formal error message
                else:
                    responses[key] = f"Content generation failed: {str(e)}"
                    break  # Stop retrying for non-quota errors

    return {
        "state": state,
        "district": district,
        "season": season,
        "crop": crop,
        "crop_year": datetime.now().year,
        "area": area,
        "predicted_yield": predicted_yield,
        "total_yield": total_production,
        "soil_recommendations": responses.get('soil', '⚠️ AI-generated content is unavailable at the moment.'),
        "pest_management": responses.get('pest', '⚠️ AI-generated content is unavailable at the moment.'),
        "irrigation_strategies": responses.get('irrigation', '⚠️ AI-generated content is unavailable at the moment.'),
        "best_farming_practices": responses.get('practices', '⚠️ AI-generated content is unavailable at the moment.'),
        "climate_impact": responses.get('climate', '⚠️ AI-generated content is unavailable at the moment.'),
        "market_trends": responses.get('market', '⚠️ AI-generated content is unavailable at the moment.'),
        "government_schemes": responses.get('schemes', '⚠️ AI-generated content is unavailable at the moment.')
    }

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Get form data
        state = request.form.get('State_Name')
        district = request.form.get('District_Name')
        season = request.form.get('Season')
        crop = request.form.get('Crop')
        crop_year = int(request.form.get('Crop_Year'))
        area = float(request.form.get('Area'))
        annual_rainfall = float(request.form.get('annual_rainfall'))

        # Get predicted yield from model
        data = CustomData(
            State_Name=state,
            District_Name=district,
            Season=season,
            Crop=crop,
            Crop_Year=crop_year,
            Area=area,
            annual_rainfall=annual_rainfall
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        predicted_yield = predict_pipeline.predict(pred_df)[0]
        
        # Generate AI content with actual yield
        report_data = generate_ai_content(state, district, season, crop, area, predicted_yield)
        
        return render_template('report.html', **report_data)

    except Exception as e:
        return render_template('report.html', error=str(e))

@app.route('/api/generate_report', methods=['POST'])
def api_generate_report():
    try:
        data = request.json
        state = data.get('State_Name')
        district = data.get('District_Name')
        season = data.get('Season')
        crop = data.get('Crop')
        crop_year = int(data.get('Crop_Year'))
        area = float(data.get('Area'))
        annual_rainfall = float(data.get('annual_rainfall'))

        # Get predicted yield from model
        data = CustomData(
            State_Name=state,
            District_Name=district,
            Season=season,
            Crop=crop,
            Crop_Year=crop_year,
            Area=area,
            annual_rainfall=annual_rainfall
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        predicted_yield = predict_pipeline.predict(pred_df)[0]
        
        # Generate AI content with actual yield
        report_data = generate_ai_content(state, district, season, crop, area, predicted_yield)
        return jsonify(report_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True) 