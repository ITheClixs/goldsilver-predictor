import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indicators import get_feature_columns

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        # XGBoost parameters
        self.params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def prepare_features(self, df, target_col='price', horizon=1):
        """
        Prepare features and target for XGBoost training
        
        Args:
            df: DataFrame with indicators
            target_col: Target column name
            horizon: Prediction horizon in days
        
        Returns:
            X (features), y (target)
        """
        print("Preparing XGBoost features...")
        
        # Get feature columns
        base_features, optional_features = get_feature_columns()
        
        # Select available features
        available_features = []
        for feature in base_features:
            if feature in df.columns:
                available_features.append(feature)
        
        for feature in optional_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # Add current price and returns as features
        if 'price' in df.columns:
            available_features.append('price')
        if 'returns' in df.columns:
            available_features.append('returns')
        
        # Remove duplicates
        available_features = list(set(available_features))
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Prepare features
        X = df[available_features].copy()
        
        # Create target (future price)
        y = df[target_col].shift(-horizon)  # Shift target by horizon days
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Store feature columns for later use
        self.feature_columns = available_features
        
        print(f"Prepared {len(X)} samples with {len(available_features)} features")
        return X, y
    
    def train(self, df, target_col='price', horizon=1):
        """Train XGBoost model"""
        try:
            print("Training XGBoost model...")
            
            # Prepare data
            X, y = self.prepare_features(df, target_col, horizon)
            
            if len(X) == 0:
                print("No training data available")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize XGBoost
            self.model = xgb.XGBRegressor(**self.params)
            
            # Train model
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Testing RMSE: {test_rmse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Testing R²: {test_r2:.4f}")
            
            # Feature importance
            feature_importance = self.model.feature_importances_
            feature_names = self.feature_columns
            
            print("\nTop 10 Most Important Features:")
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(importance_df.head(10))
            
            self.is_trained = True
            print("XGBoost training completed successfully!")
            return True
            
        except Exception as e:
            print(f"XGBoost training failed: {e}")
            return False
    
    def predict(self, df, horizon=1):
        """Make prediction for specified horizon"""
        if not self.is_trained or self.model is None:
            print("Model not trained")
            return None
        
        try:
            # Get the latest data point
            latest_data = df[self.feature_columns].tail(1)
            
            # Check for missing features
            missing_features = set(self.feature_columns) - set(latest_data.columns)
            if missing_features:
                print(f"Missing features: {missing_features}")
                return None
            
            # Scale features
            latest_scaled = self.scaler.transform(latest_data)
            
            # Make prediction
            prediction = self.model.predict(latest_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"XGBoost prediction failed: {e}")
            return None
    
    def predict_with_confidence(self, df, horizon=1, n_estimators_range=(50, 200)):
        """
        Make prediction with confidence interval using different n_estimators
        """
        if not self.is_trained:
            return None, None, None
        
        try:
            predictions = []
            
            # Make predictions with different numbers of trees
            for n_est in range(n_estimators_range[0], n_estimators_range[1] + 1, 25):
                temp_model = xgb.XGBRegressor(**{**self.params, 'n_estimators': n_est})
                
                # Use same training data (simplified approach)
                X, y = self.prepare_features(df, 'price', horizon)
                X_scaled = self.scaler.transform(X)
                temp_model.fit(X_scaled, y, verbose=False)
                
                # Predict
                latest_data = df[self.feature_columns].tail(1)
                latest_scaled = self.scaler.transform(latest_data)
                pred = temp_model.predict(latest_scaled)[0]
                predictions.append(pred)
            
            # Calculate statistics
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # 95% confidence interval
            ci_lower = mean_pred - 1.96 * std_pred
            ci_upper = mean_pred + 1.96 * std_pred
            
            return mean_pred, ci_lower, ci_upper
            
        except Exception as e:
            print(f"Confidence prediction failed: {e}")
            return None, None, None
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained or self.model is None:
            print("No trained model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            model_path = filepath.replace('.pkl', '_model.json')
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            
            self.model.save_model(model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'params': self.params
            }
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            joblib.dump(metadata, metadata_path)
            
            print(f"XGBoost model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to save XGBoost model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            model_path = filepath.replace('.pkl', '_model.json')
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            
            # Check if files exist
            if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
                print(f"Model files not found")
                return False
            
            # Load model
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata = joblib.load(metadata_path)
            self.feature_columns = metadata['feature_columns']
            self.is_trained = metadata['is_trained']
            self.params = metadata['params']
            
            print(f"XGBoost model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load XGBoost model: {e}")
            return False
    
    def get_feature_importance(self):
        """Get feature importance DataFrame"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Failed to get feature importance: {e}")
            return None

if __name__ == "__main__":
    # Test XGBoost model
    print("Testing XGBoost model...")
    
    # Create sample data with indicators
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    prices = 1000 + np.cumsum(np.random.randn(1000) * 0.02) * 50
    
    df_test = pd.DataFrame({
        'price': prices,
        'returns': pd.Series(prices).pct_change(),
        'sma_7': pd.Series(prices).rolling(7).mean(),
        'sma_14': pd.Series(prices).rolling(14).mean(),
        'sma_30': pd.Series(prices).rolling(30).mean(),
        'sma_200': pd.Series(prices).rolling(200).mean(),
        'rsi': 50 + np.random.randn(1000) * 15,
        'macd': np.random.randn(1000) * 5,
        'volatility_20': np.abs(np.random.randn(1000) * 0.1),
        'momentum_10': np.random.randn(1000) * 0.05
    }, index=dates).dropna()
    
    # Add lag features
    for col in ['returns', 'rsi', 'macd']:
        df_test[f'{col}_lag1'] = df_test[col].shift(1)
        df_test[f'{col}_lag2'] = df_test[col].shift(2)
        df_test[f'{col}_lag3'] = df_test[col].shift(3)
    
    # Add ratio features
    df_test['price_sma7_ratio'] = df_test['price'] / df_test['sma_7']
    df_test['price_sma30_ratio'] = df_test['price'] / df_test['sma_30']
    df_test['price_sma200_ratio'] = df_test['price'] / df_test['sma_200']
    
    df_test = df_test.dropna()
    
    # Initialize and train model
    xgb_model = XGBoostModel()
    
    success = xgb_model.train(df_test, horizon=5)
    if success:
        print("Training successful!")
        
        # Test prediction
        prediction = xgb_model.predict(df_test, horizon=5)
        print(f"5-day prediction: {prediction}")
        
        # Test confidence prediction
        mean_pred, ci_lower, ci_upper = xgb_model.predict_with_confidence(df_test, horizon=5)
        print(f"Confidence prediction: {mean_pred:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        # Test save/load
        xgb_model.save_model('models/test_xgb.pkl')
        
        # Load and test
        new_model = XGBoostModel()
        if new_model.load_model('models/test_xgb.pkl'):
            prediction2 = new_model.predict(df_test, horizon=5)
            print(f"Loaded model prediction: {prediction2}")
        
        # Show feature importance
        importance = xgb_model.get_feature_importance()
        if importance is not None:
            print("\nFeature Importance:")
            print(importance.head(10))
    else:
        print("Training failed!")