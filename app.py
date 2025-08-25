from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetch_data import get_current_price, fetch_historical_data
from data.indicators import calculate_indicators
from models.ensemble import EnsemblePredictor

app = Flask(__name__)

# Global predictor instance
predictor = EnsemblePredictor()

def get_trading_signal(return_pct):
    """Convert return percentage to trading signal"""
    if return_pct > 3.0:
        return "STRONG BUY", "#23ac43"
    elif return_pct > 1.0:
        return "BUY", "#20c997" 
    elif return_pct >= -1.0:
        return "HOLD", "#ffc107"
    else:
        return "SELL", "#dc3545"

@app.route('/')
def index():
    """Homepage with current prices and prediction form"""
    try:
        # Get current prices
        gold_price = get_current_price('GC=F')  # Gold futures
        silver_price = get_current_price('SI=F')  # Silver futures
        
        return render_template('index.html', 
                             gold_price=gold_price, 
                             silver_price=silver_price)
    except Exception as e:
        print(f"Error getting current prices: {e}")
        return render_template('index.html', 
                             gold_price="N/A", 
                             silver_price="N/A")

@app.route('/predict', methods=['POST'])
def predict():
    """Make price prediction and return trading signal"""
    try:
        # Get form data
        commodity = request.form.get('commodity')
        horizon = int(request.form.get('horizon'))
        
        # Map commodity to symbol
        symbol_map = {
            'gold': 'GC=F',
            'silver': 'SI=F'
        }
        
        symbol = symbol_map.get(commodity)
        if not symbol:
            return jsonify({'error': 'Invalid commodity selected'})
        
        # Get current price
        current_price = get_current_price(symbol)
        if current_price == "N/A":
            return jsonify({'error': 'Unable to fetch current price'})
        
        # Ensure models are trained
        if not predictor.is_trained():
            print("Training models...")
            success = predictor.train_models(commodity)
            if not success:
                return jsonify({'error': 'Failed to train prediction models'})
        
        # Make prediction
        predicted_price = predictor.predict(commodity, horizon)
        
        if predicted_price is None:
            return jsonify({'error': 'Prediction failed'})
        
        # Calculate return and trading signal
        return_pct = ((predicted_price - current_price) / current_price) * 100
        signal, color = get_trading_signal(return_pct)
        
        # Convert to per gram if needed (futures are typically per ounce)
        current_price_gram = current_price / 31.1035  # Troy ounce to gram
        predicted_price_gram = predicted_price / 31.1035
        
        result = {
            'commodity': commodity.title(),
            'current_price': round(current_price_gram, 2),
            'predicted_price': round(predicted_price_gram, 2),
            'horizon': horizon,
            'return_pct': round(return_pct, 2),
            'signal': signal,
            'signal_color': color,
            'success': True
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain models with latest data"""
    try:
        commodity = request.form.get('commodity', 'gold')
        success = predictor.train_models(commodity, force_retrain=True)
        
        if success:
            return jsonify({'success': True, 'message': 'Models retrained successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to retrain models'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Retraining failed: {str(e)}'})

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting Gold & Silver Price Forecasting App...")
    print("Visit http://localhost:5000 to use the application")
    
    app.run(debug=True, host='localhost', port=5000)