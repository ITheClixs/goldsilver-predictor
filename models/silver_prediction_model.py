import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
from data.indicators import calculate_indicators

class SilverPredictionModel:
    def __init__(self, sequence_length=45, hidden_size=75, num_layers=2, 
                 learning_rate=0.001, num_epochs=120):
        self.commodity = 'silver'
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0]) # Target is the next return
        return np.array(X), np.array(y)

    def prepare_data(self, df):
        df['returns'] = df['price'].pct_change()
        df = df.dropna()

        # Using a wider and more sensitive range of indicators for silver
        feature_cols = [
            'returns', 'sma_7', 'sma_14', 'sma_30', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'volatility_20', 'volatility_60', 'momentum_10', 'momentum_20',
            'price_sma7_ratio', 'price_sma30_ratio', 'volume_ratio', 'usd_momentum', 'price_usd_ratio'
        ]
        available_features = [col for col in feature_cols if col in df.columns]
        
        data = df[available_features].values
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        
        X, y = self.create_sequences(scaled_data)
        return X, y, scaler, available_features

    def train(self, force_retrain=False):
        from data.fetch_data import prepare_dataset
        
        model_path = f'models/{self.commodity}_lstm_returns.pth'
        if os.path.exists(model_path) and not force_retrain:
            self.load_model()
            return True

        df = prepare_dataset(self.commodity)
        df_with_indicators = calculate_indicators(df)
        
        X, y, scaler, feature_cols = self.prepare_data(df_with_indicators)
        self.scaler = scaler
        
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)

        model = LSTMPredictor(input_size=X.shape[2], hidden_size=self.hidden_size, num_layers=self.num_layers)
        model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            model.train()
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        
        self.model = model
        self.save_model()
        self.is_trained = True
        return True

    def predict(self, horizon=1):
        from data.fetch_data import prepare_dataset

        if not self.is_trained:
            self.train()

        model = self.model
        scaler = self.scaler

        df = prepare_dataset(self.commodity)
        df_with_indicators = calculate_indicators(df)
        
        df_with_indicators['returns'] = df_with_indicators['price'].pct_change()
        df_with_indicators = df_with_indicators.dropna()

        feature_cols = [
            'returns', 'sma_7', 'sma_14', 'sma_30', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'volatility_20', 'volatility_60', 'momentum_10', 'momentum_20',
            'price_sma7_ratio', 'price_sma30_ratio', 'volume_ratio', 'usd_momentum', 'price_usd_ratio'
        ]
        available_features = [col for col in feature_cols if col in df_with_indicators.columns]
        
        last_data = df_with_indicators[available_features].tail(self.sequence_length).values
        last_data_scaled = scaler.transform(last_data)
        
        model.eval()
        
        predicted_returns = []
        current_price = df_with_indicators['price'].iloc[-1]
        
        with torch.no_grad():
            for _ in range(horizon):
                X_input = torch.FloatTensor(last_data_scaled.reshape(1, self.sequence_length, -1)).to(self.device)
                predicted_return_scaled = model(X_input).cpu().numpy().flatten()
                
                dummy_data = np.zeros((1, len(available_features)))
                dummy_data[0, 0] = predicted_return_scaled[0]
                predicted_return = scaler.inverse_transform(dummy_data)[0, 0]
                
                predicted_returns.append(predicted_return)
                
                new_row = last_data[-1, :].copy()
                new_row[0] = predicted_return
                last_data = np.vstack([last_data[1:], new_row])
                last_data_scaled = scaler.transform(last_data)

        predicted_price = current_price
        for r in predicted_returns:
            predicted_price = predicted_price * (1 + r)
            
        return predicted_price

    def save_model(self):
        filepath = f'models/{self.commodity}_lstm_returns.pth'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
        }
        torch.save(checkpoint, filepath)

    def load_model(self):
        filepath = f'models/{self.commodity}_lstm_returns.pth'
        checkpoint = torch.load(filepath, weights_only=False)
        self.scaler = checkpoint['scaler']
        
        input_size = self.scaler.scale_.shape[0]
        
        model = LSTMPredictor(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        self.model = model
        self.is_trained = True

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out