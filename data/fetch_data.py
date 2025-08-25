// Placeholder for static/script.js
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time

def fetch_historical_data(symbol, period="10y"):
    """
    Fetch historical price data from Yahoo Finance
    
    Args:
        symbol: Yahoo Finance symbol (e.g., 'GC=F' for gold, 'SI=F' for silver)
        period: Time period ('1y', '2y', '5y', '10y', 'max')
    
    Returns:
        pandas.DataFrame with OHLCV data
    """
    try:
        print(f"Fetching historical data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
        
        # Clean data
        data = data.dropna()
        data.index = pd.to_datetime(data.index)
        
        print(f"Successfully fetched {len(data)} records for {symbol}")
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_current_price(symbol):
    """
    Get current/latest price for a symbol
    
    Args:
        symbol: Yahoo Finance symbol
    
    Returns:
        float: Current price or "N/A" if failed
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Try to get current price from info
        info = ticker.info
        if 'regularMarketPrice' in info:
            return float(info['regularMarketPrice'])
        elif 'previousClose' in info:
            return float(info['previousClose'])
        
        # Fallback: get latest price from history
        hist = ticker.history(period="5d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        return "N/A"
        
    except Exception as e:
        print(f"Error getting current price for {symbol}: {e}")
        return "N/A"

def fetch_usd_index():
    """
    Fetch USD Index (DXY) data
    
    Returns:
        pandas.DataFrame with USD Index data
    """
    try:
        print("Fetching USD Index (DXY) data...")
        
        ticker = yf.Ticker("DX-Y.NYB")
        data = ticker.history(period="10y")
        
        if data.empty:
            print("No DXY data found")
            return None
        
        data = data.dropna()
        print(f"Successfully fetched {len(data)} DXY records")
        return data
        
    except Exception as e:
        print(f"Error fetching DXY data: {e}")
        return None

def prepare_dataset(commodity='gold'):
    """
    Prepare complete dataset with prices, USD index, and technical indicators
    
    Args:
        commodity: 'gold' or 'silver'
    
    Returns:
        pandas.DataFrame with all features
    """
    try:
        # Symbol mapping
        symbol_map = {
            'gold': 'GC=F',
            'silver': 'SI=F'
        }
        
        symbol = symbol_map.get(commodity)
        if not symbol:
            raise ValueError(f"Unknown commodity: {commodity}")
        
        # Fetch commodity data
        commodity_data = fetch_historical_data(symbol)
        if commodity_data is None:
            return None
        
        # Fetch USD Index
        usd_data = fetch_usd_index()
        
        # Create base dataset
        df = pd.DataFrame(index=commodity_data.index)
        df['price'] = commodity_data['Close']
        df['volume'] = commodity_data['Volume']
        df['high'] = commodity_data['High']
        df['low'] = commodity_data['Low']
        df['open'] = commodity_data['Open']
        
        # Add USD Index if available
        if usd_data is not None:
            # Align USD data with commodity data
            usd_aligned = usd_data['Close'].reindex(df.index, method='ffill')
            df['usd_index'] = usd_aligned
        else:
            df['usd_index'] = 100  # Fallback constant
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        print(f"Prepared dataset with {len(df)} records for {commodity}")
        return df
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return None

def save_data(df, filename):
    """Save DataFrame to CSV"""
    try:
        df.to_csv(filename)
        print(f"Data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_data(filename):
    """Load DataFrame from CSV"""
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Data loaded from {filename}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Test data fetching
    print("Testing data fetching...")
    
    # Test current prices
    gold_price = get_current_price('GC=F')
    silver_price = get_current_price('SI=F')
    
    print(f"Current Gold Price: ${gold_price}")
    print(f"Current Silver Price: ${silver_price}")
    
    # Test dataset preparation
    gold_data = prepare_dataset('gold')
    if gold_data is not None:
        print(f"Gold dataset shape: {gold_data.shape}")
        print(gold_data.head())
        
        # Save for later use
        save_data(gold_data, 'data/gold_data.csv')
    
    silver_data = prepare_dataset('silver')
    if silver_data is not None:
        print(f"Silver dataset shape: {silver_data.shape}")
        print(silver_data.head())
        
        # Save for later use
        save_data(silver_data, 'data/silver_data.csv')