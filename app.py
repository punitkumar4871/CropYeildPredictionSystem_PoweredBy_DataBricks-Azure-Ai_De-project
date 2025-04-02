from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import google.generativeai as genai
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Configure Gemini AI API Key
genai.configure(api_key="YAIzaSyC8tuqLoWlAMrz90XmNkXBL36ErDppMD4I")

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
            # Collecting user input
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

def generate_farming_report(state, district, season, crop, crop_year, area, predicted_yield):
    """
    Generate AI-based farming report with updated content.
    """
    total_yield = round(predicted_yield * area, 2)

    # AI-Generated Recommendations
    soil_recommendations = f"""
    Based on soil conditions in {district}, farmers should:
    - **Ideal pH:** 6.0 - 7.5 for optimal {crop} growth.
    - **Fertilizer:** Urea (50 kg/ha), SSP (50 kg/ha), MOP (25 kg/ha), and organic manure.
    - **Soil Treatment:** Lime for acidic soil, gypsum for saline soil.
    - **Crop Rotation:** Rotate {crop} with pulses to maintain soil fertility.
    """

    pest_management = f"""
    Common threats for {crop} farming in {district}:
    - **Pests:** Stem Borers, Aphids, and Brown Planthoppers.
    - **Natural Control:** Introduce ladybugs and neem-based pesticides.
    - **Chemical Control:** Use recommended insecticides in controlled amounts.
    - **Disease Prevention:** Ensure proper spacing and crop rotation.
    """

    irrigation_strategies = f"""
    Effective water management for {crop} farming:
    - **Drip Irrigation:** Helps conserve water and promotes root health.
    - **Rainwater Harvesting:** Store excess rainwater for dry periods.
    - **Drainage System:** Prevents waterlogging and improves yield.
    """

    best_farming_practices = f"""
    Key strategies to maximize {crop} yield:
    - **Seed Selection:** Use disease-resistant and high-yield varieties.
    - **Land Preparation:** Deep plowing and leveling for uniform water distribution.
    - **Weed Control:** Manual weeding or targeted herbicides every 15-20 days.
    - **Timely Harvesting:** Prevent losses by harvesting at optimal moisture levels.
    """

    climate_impact = f"""
    Climate change affects {crop} farming by:
    - **Temperature Changes:** Rising temperatures may reduce yields.
    - **Unpredictable Rainfall:** Can cause droughts or floods.
    - **Solution:** Use climate-resistant crop varieties and adaptive farming techniques.
    """

    market_trends = f"""
    - **Current Market Price of {crop}:** Prices fluctuate based on demand and supply.
    - **Export Trends:** Demand for {crop} has been rising in international markets.
    - **Tip for Farmers:** Monitor market rates before selling produce.
    """

    government_schemes = f"""
    - **PM-KISAN:** Direct financial support for farmers.
    - **Fasal Bima Yojana:** Crop insurance to protect against losses.
    - **Subsidized Seeds & Equipment:** Farmers can get discounts on high-quality seeds.
    """

    return {
        "state": state,
        "district": district,
        "season": season,
        "crop": crop,
        "crop_year": crop_year,
        "area": area,
        "predicted_yield": predicted_yield,
        "total_yield": total_yield,
        "soil_recommendations": soil_recommendations,
        "pest_management": pest_management,
        "irrigation_strategies": irrigation_strategies,
        "best_farming_practices": best_farming_practices,
        "climate_impact": climate_impact,
        "market_trends": market_trends,
        "government_schemes": government_schemes
    }

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Collect user inputs
        state = request.form.get('State_Name')
        district = request.form.get('District_Name')
        season = request.form.get('Season')
        crop = request.form.get('Crop')
        crop_year = int(request.form.get('Crop_Year'))
        area = float(request.form.get('Area'))

        # Get predicted yield from the model
        data = CustomData(state, district, season, crop, crop_year, area, annual_rainfall=0)
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        predicted_yield = results[0]

        # Generate report with the correct yield
        report_data = generate_farming_report(state, district, season, crop, crop_year, area, predicted_yield)

        return render_template('report.html', **report_data)

    except Exception as e:
        return render_template('report.html', report=f"Error: {str(e)}")

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

        # Get predicted yield
        data = CustomData(state, district, season, crop, crop_year, area, annual_rainfall=0)
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        predicted_yield = results[0]

        # Generate report with correct yield
        report_data = generate_farming_report(state, district, season, crop, crop_year, area, predicted_yield)
        return jsonify(report_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)