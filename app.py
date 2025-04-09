from flask import Flask, request, render_template, jsonify, session, redirect, url_for, copy_current_request_context
import numpy as np
import pandas as pd
import google.generativeai as genai
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import time
from datetime import datetime
import threading
from threading import Lock
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Configure Gemini AI API Key
genai.configure(api_key="AIzaSyBPA8qtCAAttKztC_2s66u8dANYDOb_sKc")

application = Flask(__name__)
app = application
app.secret_key = 'abcd123456789'  # Needed for session

# Thread-safe storage for report data
report_data_cache = {}
cache_lock = Lock()

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
    """Generate comprehensive farming recommendations using Gemini AI with fallbacks"""
    total_production = round(predicted_yield * area, 2)

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

    # Fallback responses for each section
    fallbacks = {
        'soil': f"For {crop}, ensure soil is well-drained and rich in organic matter. Test soil pH and adjust with lime or sulfur as needed (ideal range: 6.0-7.0). Add compost or manure to improve fertility. Avoid waterlogging and consider crop-specific nutrient needs like nitrogen, phosphorus, and potassium.",
        'pest': f"Common pests for {crop} may include aphids, beetles, and caterpillars, while diseases like fungal infections or blight can occur. Monitor crops regularly, use organic pesticides if possible, and remove affected plants to prevent spread. Consult local experts for region-specific threats.",
        'irrigation': f"Irrigate {crop} based on {season} conditions—typically 1-2 inches of water weekly. Use drip or sprinkler systems for efficiency. Avoid overwatering to prevent root rot. Adjust frequency during dry spells or heavy rains, ensuring consistent soil moisture.",
        'practices': f"For {crop}, rotate crops yearly to maintain soil health, use quality seeds, and plant at optimal spacing. Weed regularly, apply mulch to retain moisture, and time planting with {season} weather patterns. Monitor growth and adjust care as needed.",
        'climate': f"In {district}, {season} weather may affect {crop} with temperature swings or rainfall changes. Protect young plants from extreme heat or cold, and adjust planting schedules if unpredictable patterns occur. Adequate shade or drainage can mitigate impacts.",
        'market': f"Market trends for {crop} in {state} depend on supply and demand. Prices may rise with low regional production or fall during surpluses. Check local markets or co-ops for current rates and consider selling at peak demand times.",
        'schemes': f"Farmers in {district}, {state} growing {crop} may access subsidies for seeds, fertilizers, or equipment. Look into national schemes like PM-KISAN or state-specific programs. Contact local agriculture offices for eligibility and application details."
    }

    responses = {}
    for key, prompt in prompts.items():
        for attempt in range(3):  # Retry up to 3 times
            try:
                time.sleep(2)  # Delay between requests
                response = model.generate_content(prompt)
                responses[key] = response.text if response.text else fallbacks[key]
                break
            except Exception as e:
                error_msg = str(e).lower()
                if "quota exceeded" in error_msg or "429" in error_msg or attempt == 2:
                    # Use fallback content if quota exceeded or all retries fail
                    responses[key] = fallbacks[key]
                    break
                # Otherwise, retry on other errors

    return {
        "state": state,
        "district": district,
        "season": season,
        "crop": crop,
        "crop_year": datetime.now().year,
        "area": area,
        "predicted_yield": predicted_yield,
        "total_yield": total_production,
        "soil_recommendations": responses.get('soil'),
        "pest_management": responses.get('pest'),
        "irrigation_strategies": responses.get('irrigation'),
        "best_farming_practices": responses.get('practices'),
        "climate_impact": responses.get('climate'),
        "market_trends": responses.get('market'),
        "government_schemes": responses.get('schemes')
    }

def generate_report_in_background(form_data, session_id):
    @copy_current_request_context
    def generate_and_store():
        try:
            state = form_data.get('State_Name')
            district = form_data.get('District_Name')
            season = form_data.get('Season')
            crop = form_data.get('Crop')
            crop_year = int(form_data.get('Crop_Year'))
            area = float(form_data.get('Area'))
            annual_rainfall = float(form_data.get('annual_rainfall'))

            # Get predicted yield
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
            
            # Generate AI content
            report_data = generate_ai_content(state, district, season, crop, area, predicted_yield)
            
            # Store in cache
            with cache_lock:
                report_data_cache[session_id] = report_data

        except Exception as e:
            with cache_lock:
                report_data_cache[session_id] = {"error": str(e)}
    
    # Start the generation
    thread = threading.Thread(target=generate_and_store)
    thread.start()

@app.route('/generate_report', methods=['POST'])
def show_loader():
    # Create unique session ID
    session_id = str(time.time())
    session['session_id'] = session_id
    
    # Store form data in session
    session['form_data'] = request.form.to_dict()
    
    # Start background generation
    generate_report_in_background(request.form, session_id)
    
    # Show loader page
    return render_template('loader.html', session_id=session_id)

@app.route('/check_report')
def check_report():
    session_id = request.args.get('session_id', '')
    with cache_lock:
        ready = session_id in report_data_cache
    return jsonify({"ready": ready})

@app.route('/report')
def show_report():
    session_id = session.get('session_id', '')
    
    # Check if report is ready (with timeout)
    start_time = time.time()
    while time.time() - start_time < 60:  # 60 second timeout
        with cache_lock:
            if session_id in report_data_cache:
                report_data = report_data_cache.get(session_id, {})
                if 'error' in report_data:
                    return render_template('report.html', error=report_data['error'])
                return render_template('report.html', session_id=session_id, **report_data)
        time.sleep(0.5)  # Check every 0.5 seconds
    
    # If timeout reached
    return render_template('report.html', 
                         error="Report generation is taking longer than expected. Please try again later.")

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
        
        report_data = generate_ai_content(state, district, season, crop, area, predicted_yield)
        return jsonify(report_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/download_report/<session_id>')
def download_report(session_id):
    try:
        with cache_lock:
            report_data = report_data_cache.get(session_id, None)
            if not report_data or 'error' in report_data:
                return "Report not found or generation failed", 404

        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add report content to PDF
        story.append(Paragraph("Comprehensive Farming Report", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        fields = [
            f"State: {report_data['state']}",
            f"District: {report_data['district']}",
            f"Season: {report_data['season']}",
            f"Crop: {report_data['crop']}",
            f"Crop Year: {report_data['crop_year']}",
            f"Land Area: {report_data['area']} hectares",
            f"Predicted Yield per hectare: {report_data['predicted_yield']} tons",
            f"Total Estimated Yield: {report_data['total_yield']} tons"
        ]
        
        for field in fields:
            story.append(Paragraph(field, styles['Normal']))
            story.append(Spacer(1, 6))

        sections = [
            ("Soil & Fertilizer Recommendations", "soil_recommendations"),
            ("Pest & Disease Management", "pest_management"),
            ("Water & Irrigation Strategies", "irrigation_strategies"),
            ("Best Farming Practices", "best_farming_practices"),
            ("Climate Impact on Farming", "climate_impact"),
            ("Market Trends & Pricing", "market_trends"),
            ("Government Schemes & Subsidies", "government_schemes")
        ]
        
        for title, key in sections:
            story.append(Paragraph(title, styles['Heading2']))
            story.append(Paragraph(report_data[key], styles['Normal']))
            story.append(Spacer(1, 12))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Send file as download
        return app.response_class(
            buffer.getvalue(),
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment;filename={report_data["crop"]}_report_{session_id}.pdf'}
        )
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)