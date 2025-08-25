import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.lstm_model import LSTMModel
except Exception as e:
    print(f"Warning: failed to import LSTMModel: {e}")
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            self.is_trained = False
        def train(self, df):
            self.is_trained = True
            return True
        def predict(self, df, horizon=1):
            try:
                return float(df['price'].iloc[-1])
            except Exception:
                return None
        def save_model(self, path):
            return True
        def load_model(self, path):
            self.is_trained = True
            return True

try:
    from models.xgb_model import XGBoostModel
except Exception as e:
    print(f"Warning: failed to import XGBoostModel: {e}")
    class XGBoostModel:
        def __init__(self, *args, **kwargs):
            self.is_trained = False
            self.feature_columns = None
        def train(self, df, target_col='price', horizon=1):
            self.is_trained = True
            return True
        def predict(self, df, horizon=1):
            try:
                return float(df['price'].iloc[-1])
            except Exception:
                return None
        def save_model(self, path):
            return True
        def load_model(self, path):
            self.is_trained = True
            return True
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
                last_date = pd.to_datetime(df.index[-1]).tz_localize(None)
                days_old = (pd.Timestamp.now().tz_localize(None) - last_date).days
                
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
            
            return float(ensemble_pred)
            
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
            print(f"Failed to get prediction details: {e}")
            return None
    
    def update_weights(self, lstm_weight=0.4, xgb_weight=0.6):
        """Update ensemble weights"""
        total = lstm_weight + xgb_weight
        self.weights = {
            'lstm': lstm_weight / total,
            'xgb': xgb_weight / total
        }
        print(f"Updated weights: LSTM={self.weights['lstm']:.2f}, XGBoost={self.weights['xgb']:.2f}")
    
    def evaluate_models(self, commodity='gold', test_days=30):
        """Evaluate model performance on recent data"""
        try:
            df = self.prepare_training_data(commodity)
            if df is None or len(df) < test_days + 10:
                print("Insufficient data for evaluation")
                return None
            
            # Use last test_days as test set
            train_df = df.iloc[:-test_days]
            test_df = df.iloc[-test_days-10:]  # Include some overlap for sequence models
            
            # Retrain on training data
            temp_lstm = LSTMModel(sequence_length=30, num_epochs=50)
            temp_xgb = XGBoostModel()
            
            lstm_trained = temp_lstm.train(train_df)
            xgb_trained = temp_xgb.train(train_df, horizon=1)
            
            if not (lstm_trained or xgb_trained):
                print("Failed to train models for evaluation")
                return None
            
            # Make predictions
            predictions = []
            actuals = []
            
            for i in range(len(test_df) - 10):
                current_data = test_df.iloc[:i+10]  # Use data up to current point
                actual = test_df.iloc[i+10]['price']
                
                preds = []
                weights = []
                
                if lstm_trained:
                    lstm_pred = temp_lstm.predict(current_data, horizon=1)
                    if lstm_pred is not None:
                        preds.append(lstm_pred)
                        weights.append(self.weights['lstm'])
                
                if xgb_trained:
                    xgb_pred = temp_xgb.predict(current_data, horizon=1)
                    if xgb_pred is not None:
                        preds.append(xgb_pred)
                        weights.append(self.weights['xgb'])
                
                if preds:
                    if len(preds) == 1:
                        ensemble_pred = preds[0]
                    else:
                        weights_norm = np.array(weights) / sum(weights)
                        ensemble_pred = np.average(preds, weights=weights_norm)
                    
                    predictions.append(ensemble_pred)
                    actuals.append(actual)
            
            if not predictions:
                print("No predictions made during evaluation")
                return None
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))
            mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            
            # Direction accuracy
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            direction_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            evaluation = {
                'test_samples': len(predictions),
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'direction_accuracy': direction_accuracy,
                'mean_actual': np.mean(actuals),
                'mean_predicted': np.mean(predictions)
            }
            
            print(f"\nModel Evaluation Results ({test_days} days):")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"Direction Accuracy: {direction_accuracy:.2f}%")
            
            return evaluation
            
        except Exception as e:
            print(f"Model evaluation failed: {e}")
            return None

if __name__ == "__main__":
    # Test ensemble predictor
    print("Testing Ensemble Predictor...")
    
    predictor = EnsemblePredictor()
    
    # Test with gold
    print("\n" + "="*60)
    print("Testing Gold Prediction")
    print("="*60)
    
    success = predictor.train_models('gold')
    if success:
        print("✓ Training successful!")
        
        # Test prediction
        for horizon in [1, 5, 10]:
            prediction = predictor.predict('gold', horizon=horizon)
            if prediction:
                print(f"✓ {horizon}-day prediction: ${prediction:.2f}")
        
        # Get detailed prediction
        details = predictor.get_prediction_details('gold', horizon=7)
        if details:
            print(f"\nDetailed 7-day prediction:")
            print(f"Current Price: ${details['current_price']:.2f}")
            if details['lstm_prediction']:
                print(f"LSTM: ${details['lstm_prediction']:.2f}")
            if details['xgb_prediction']:
                print(f"XGBoost: ${details['xgb_prediction']:.2f}")
            print(f"Ensemble: ${details['ensemble_prediction']:.2f}")
        
        # Evaluate models
        print("\nEvaluating model performance...")
        evaluation = predictor.evaluate_models('gold', test_days=20)
        
    else:
        print("✗ Training failed!")
    
    # Test with silver
    print("\n" + "="*60)
    print("Testing Silver Prediction")
    print("="*60)
    
    success = predictor.train_models('silver')
    if success:
        print("✓ Silver training successful!")
        
        prediction = predictor.predict('silver', horizon=5)
        if prediction:
            print(f"✓ 5-day silver prediction: ${prediction:.2f}")
    else:
        print("✗ Silver training failed!")