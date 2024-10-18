from flask import Flask, render_template, request, jsonify
from forecasting import load_and_preprocess_data, filter_data_by_date, prepare_hourly_data, sarima_forecast, plot_forecast
from billing import calculate_energy_bill, get_billing_rates
import pandas as pd
import io
import base64
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global variable to store the JSON data
json_data = {}

@app.route('/')
def home():
    api_url = "https://render-ivuy.onrender.com/data"
    data = load_and_preprocess_data(api_url)
    
    input_data = data.head(100).reset_index().to_dict(orient='records')  # Prepare data for HTML rendering
    
    return render_template('index.html', input_data=input_data)

@app.route('/forecast', methods=['POST'])
def forecast():
    global json_data  # Access the global variable
    
    api_url = "https://render-ivuy.onrender.com/data"

    # Get user input
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    forecast_hours = int(request.form['forecast_hours'])

    # Load and preprocess data
    data = load_and_preprocess_data(api_url)
    filtered_data = filter_data_by_date(data, start_date, end_date)

    # Prepare hourly data for forecasting
    hourly_kvah_diff, hourly_kvah = prepare_hourly_data(filtered_data)

    # Forecasting using SARIMA
    forecast_df, _ = sarima_forecast(hourly_kvah_diff, (1, 0, 1), (1, 1, 1, 24), forecast_hours)

    # Billing calculation
    rates = get_billing_rates()
    total_hours = len(forecast_df)
    charges = calculate_energy_bill(forecast_df, rates, total_hours)

    # Store hourly_kvah values
    actual_hourly_kvah = hourly_kvah.reset_index()
    actual_hourly_kvah.columns = ['DateTime', 'Actual_kVAh']
    actual_hourly_kvah['DateTime'] = actual_hourly_kvah['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Create a structured JSON object
    json_data = {
        'actual_hourly_kVAh': actual_hourly_kvah.to_dict(orient='records'),
        'forecasted_kVAh': forecast_df[['Date_Hourly', 'Forecasted_kVah']].rename(columns={'Date_Hourly': 'DateTime'}).copy()
    }

    # Convert forecasted DataFrame to dictionary and format DateTime
    json_data['forecasted_kVAh']['DateTime'] = json_data['forecasted_kVAh']['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    json_data['forecasted_kVAh'] = json_data['forecasted_kVAh'].to_dict(orient='records')

    return render_template('results.html', charges=charges)

@app.route('/api/forecast-data', methods=['GET'])
def get_forecast_data():
    return jsonify(json_data)  # Send the JSON data as response

if __name__ == '__main__':
    app.run()
