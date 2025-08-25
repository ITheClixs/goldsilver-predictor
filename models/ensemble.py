import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMModel
from models.xgb_model import XGBoostModel
from data.fetch_data import prepare_dataset, load_data, save_data
from data.indicators import calculate_indicators

class EnsemblePredictor:
    def __init__(self):
        self.lstm_model = LSTMModel(sequence_length=30, num_epochs=100)
        self.xgb_model = XGBoostModel()
        self.weights = {'lstm': 0.4, 'xgb': 0.6}  # XGBoost typically more stable
        self.trained_commodities = set()
        
    def is_trained(self, commodity='gold'):
        """Check if models are trained for the commodity"""
        return commodity in self.trained_commodities
    
    def prepare_training_data(self, commodity='gold'):
        """Prepare training data with indicators"""
        print(f"Preparing training data for {commodity}...")
        
        # Try to load existing data first
        data_file = f'data/{commodity}_data.csv'
        
        if os.path.exists(data_file):
            print(f"Loading existing data from {data_file}")
            df = load_data(data_file)
            
            # Check if data is recent (within last 7 days)
            if df is not None and not df.empty:
                last_date = pd.to_datetime(df.index[-1])
                days_old = (pd.Timestamp.now() - last_date).days
                
                if days_old <= 7:
                    print(f"Using cached data (last updated {days_old} days ago)")
                    return calculate_indicators(df)
        
        # Fetch fresh data
        print("Fetching fresh data...")
        df = prepare_dataset(commodity)
        
        if df is None or df.empty:
            print(f"Failed to prepare dataset for {commodity}")
            return None
        
        # Calculate indicators
        df_with_indicators = calculate_indicators(df)
        
        # Save for future use
        if save_data(df_with_indicators, data_file):
            print(f"Data saved to {data_file}")
        
        return df_with_indicators
    
    def train_models(self, commodity='gold', force_retrain=False):
        """Train both LSTM and XGBoost models"""
        print(f"Training ensemble models for {commodity}...")
        
        # Check if already trained and not forcing retrain
        if not force_retrain and self.is_trained(commodity):
            print(f"Models already trained for {commodity}")
            return True
        
        try:
            # Prepare data
            df = self.prepare_training_data(commodity)
            if df is None or len(df) < 100:
                print("Insufficient training data")
                return False
            
            print(f"Training data shape: {df.shape}")
            
            # Model file paths
            lstm_path = f'models/{commodity}_lstm.pth'
            xgb_path = f'models/{commodity}_xgb.pkl'
            
            # Train LSTM model
            print("\n" + "="*50)
            print("Training LSTM Model")
            print("="*50)
            
            lstm_success = False
            if not os.path.exists(lstm_path) or force_retrain:
                lstm_success = self.lstm_model.train(df)
                if lstm_success:
                    self.lstm_model.save_model(lstm_path)
            else:
                lstm_success = self.lstm_model.load_model(lstm_path)
            
            # Train XGBoost model
            print("\n" + "="*50)
            print("Training XGBoost Model")
            print("="*50)
            
            xgb_success = False
            if not os.path.exists(xgb_path) or force_retrain:
                xgb_success = self.xgb_model.train(df, horizon=1)  # Start with 1-day horizon
                if xgb_success:
                    self.xgb_model.save_model(xgb_path)
            else:
                xgb_success = self.xgb_model.load_model(xgb_path)
            
            # Check training results
            if lstm_success and xgb_success:
                self.trained_commodities.add(commodity)
                print(f"\n✓ Ensemble training completed for {commodity}")
                return True
            elif lstm_success or xgb_success:
                # Adjust weights if only one model trained successfully
                if lstm_success and not xgb_success:
                    self.weights = {'lstm': 1.0, 'xgb': 0.0}
                    print("Only LSTM trained successfully, using LSTM only")
                elif xgb_success and not lstm_success:
                    self.weights = {'lstm': 0.0, 'xgb': 1.0}
                    print("Only XGBoost trained successfully, using XGBoost only")
                
                self.trained_commodities.add(commodity)
                return True
            else:
                print("Both models failed to train")
                return False
                
        except Exception as e:
            print(f"Ensemble training failed: {e}")
            return False
    
    def predict(self, commodity='gold', horizon=1):
        """Make ensemble prediction"""
        if not self.is_trained(commodity):
            print(f"Models not trained for {commodity}")
            return None
        
        try:
            # Load data for prediction
            df = self.prepare_training_data(commodity)
            if df is None:
                print("Failed to load prediction data")
                return None
            
            predictions = []
            weights_used = []
            
            # LSTM prediction
            lstm_pred = None
            if self.weights['lstm'] > 0:
                try:
                    # Load LSTM model if not already loaded
                    lstm_path = f'models/{commodity}_lstm.pth'
                    if not self.lstm_model.is_trained:
                        self.lstm_model.load_model(lstm_path)
                    
                    lstm_pred = self.lstm_model.predict(df, horizon)
                    if lstm_pred is not None:
                        predictions.append(lstm_pred)
                        weights_used.append(self.weights['lstm'])
                        print(f"LSTM prediction: {lstm_pred:.2f}")
                except Exception as e:
                    print(f"LSTM prediction failed: {e}")
            
            # XGBoost prediction
            xgb_pred = None
            if self.weights['xgb'] > 0:
                try:
                    # Load XGBoost model if not already loaded
                    xgb_path = f'models/{commodity}_xgb.pkl'
                    if not self.xgb_model.is_trained:
                        self.xgb_model.load_model(xgb_path)
                    
                    xgb_pred = self.xgb_model.predict(df, horizon)
                    if xgb_pred is not None:
                        predictions.append(xgb_pred)
                        weights_used.append(self.weights['xgb'])
                        print(f"XGBoost prediction: {xgb_pred:.2f}")
                except Exception as e:
                    print(f"XGBoost prediction failed: {e}")
            
            # Ensemble prediction
            if not predictions:
                print("No valid predictions from either model")
                return None
            
            if len(predictions) == 1:
                # Only one model worked
                ensemble_pred = predictions[0]
                print(f"Single model prediction: {ensemble_pred:.2f}")
            else:
                # Weighted average
                weights_normalized = np.array(weights_used) / sum(weights_used)
                ensemble_pred = np.average(predictions, weights=weights_normalized)
                print(f"Ensemble prediction (LSTM: {lstm_pred:.2f}, XGB: {xgb_pred:.2f}): {ensemble_pred:.2f}")
            
            return ensemble_pred
            
        except Exception as e:
            print(f"Ensemble prediction failed: {e}")
            return None
    
    def get_prediction_details(self, commodity='gold', horizon=1):
        """Get detailed prediction information including individual model predictions"""
        if not self.is_trained(commodity):
            return None
        
        try:
            df = self.prepare_training_data(commodity)
            if df is None:
                return None
            
            details = {
                'commodity': commodity,
                'horizon': horizon,
                'lstm_prediction': None,
                'xgb_prediction': None,
                'ensemble_prediction': None,
                'weights': self.weights.copy(),
                'current_price': df['price'].iloc[-1] if 'price' in df.columns else None
            }
            
            # Get individual predictions
            if self.weights['lstm'] > 0:
                lstm_path = f'models/{commodity}_lstm.pth'
                if not self.lstm_model.is_trained:
                    self.lstm_model.load_model(lstm_path)
                details['lstm_prediction'] = self.lstm_model.predict(df, horizon)
            
            if self.weights['xgb'] > 0:
                xgb_path = f'models/{commodity}_xgb.pkl'
                if not self.xgb_model.is_trained:
                    self.xgb_model.load_model(xgb_path)
                details['xgb_prediction'] = self.xgb_model.predict(df, horizon)
            
            # Calculate ensemble prediction
            predictions = []
            weights_used = []
            
            if details['lstm_prediction'] is not None:
                predictions.append(details['lstm_prediction'])
                weights_used.append(self.weights['lstm'])
            
            if details['xgb_prediction'] is not None:
                predictions.append(details['xgb_prediction'])
                weights_used.append(self.weights['xgb'])
            
            if predictions:
                if len(predictions) == 1:
                    details['ensemble_prediction'] = predictions[0]
                else:
                    weights_normalized = np.array(weights_used) / sum(weights_used)
                    details['ensemble_prediction'] = np.average(predictions, weights=weights_normalized)
            
            return details
            
        except Exception as e:
            print(f"Failed to get prediction details: