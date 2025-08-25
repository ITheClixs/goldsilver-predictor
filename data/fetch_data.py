import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import os

_HAS_YFINANCE = True
try:
    import yfinance as yf
except Exception:
    _HAS_YFINANCE = False

def fetch_historical_data(symbol, period="10y"):
    """
    Fetch historical price data from Yahoo Finance
    
    Args:
        symbol: Yahoo Finance symbol (e.g., 'GC=F' for gold, 'SI=F' for silver)
        period: Time period ('1y', '2y', '5y', '10y', 'max')
    
    Returns:
        pandas.DataFrame with OHLCV data
    """
    # Prefer yfinance when available, otherwise try cached CSV or synthetic data
    print(f"Fetching historical data for {symbol}...")
    symbol_map = {'GC=F': 'gold', 'SI=F': 'silver'}
    commodity = symbol_map.get(symbol, symbol.replace('=', '_').replace('/', '_'))

    if _HAS_YFINANCE:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data is None or data.empty:
                raise RuntimeError('No data from yfinance')
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            print(f"Successfully fetched {len(data)} records for {symbol} via yfinance")
            return data
        except Exception as e:
            print(f"yfinance fetch failed for {symbol}: {e}")

    # Fallback: look for cached CSV
    csv_path = f'data/{commodity}_data.csv'
    if os.path.exists(csv_path):
        try:
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"Loaded cached data from {csv_path}")
            return data
        except Exception as e:
            print(f"Failed to load cached CSV {csv_path}: {e}")

    # Last resort: generate synthetic data
    print("Generating synthetic data as fallback...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=365, freq='D')
    prices = 1800 + np.cumsum(np.random.randn(len(dates)) * 2)  # base ~1800
    data = pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.002,
        'Low': prices * 0.998,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    try:
        os.makedirs('data', exist_ok=True)
        data.to_csv(csv_path)
        print(f"Saved synthetic data to {csv_path}")
    except Exception:
        pass

    return data

def get_current_price(symbol):
    """
    Get current/latest price for a symbol
    
    Args:
        symbol: Yahoo Finance symbol
    
    Returns:
        float: Current price or "N/A" if failed
    """
    # Try yfinance first
    if _HAS_YFINANCE:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if isinstance(info, dict) and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                return float(info['regularMarketPrice'])
            elif isinstance(info, dict) and 'previousClose' in info:
                return float(info['previousClose'])

            hist = ticker.history(period="5d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            print(f"yfinance current price fetch failed for {symbol}: {e}")

    # Fallback: read latest from cached CSV
    symbol_map = {'GC=F': 'gold', 'SI=F': 'silver'}
    commodity = symbol_map.get(symbol, symbol.replace('=', '_'))
    csv_path = f'data/{commodity}_data.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if 'Close' in df.columns:
                return float(df['Close'].iloc[-1])
            if 'price' in df.columns:
                return float(df['price'].iloc[-1])
        except Exception as e:
            print(f"Failed to read current price from {csv_path}: {e}")

    return "N/A"

def fetch_usd_index():
    """
    Fetch USD Index (DXY) data
    
    Returns:
        pandas.DataFrame with USD Index data
    """
    # Try via yfinance if available
    if _HAS_YFINANCE:
        try:
            ticker = yf.Ticker("DX-Y.NYB")
            data = ticker.history(period="10y")
            if data is None or data.empty:
                return None
            data = data.dropna()
            return data
        except Exception:
            pass

    # Otherwise, return None (indicate unavailable)
    return None

def prepare_dataset(commodity='gold'):
    """
    Prepare complete dataset with prices, USD index, and technical indicators
    
    Args:
        commodity: 'gold' or 'silver'
    
    Returns:
        pandas.DataFrame with all features
    """
    # Symbol mapping
    symbol_map = {
        'gold': 'GC=F',
        'silver': 'SI=F'
    }

    symbol = symbol_map.get(commodity)
    if not symbol:
        raise ValueError(f"Unknown commodity: {commodity}")

    # Fetch or load commodity data
    commodity_data = fetch_historical_data(symbol)
    if commodity_data is None:
        return None

    usd_data = fetch_usd_index()

    df = pd.DataFrame(index=commodity_data.index)
    # Handle both yfinance and synthetic/cached formats
    if 'Close' in commodity_data.columns:
        df['price'] = commodity_data['Close']
    elif 'price' in commodity_data.columns:
        df['price'] = commodity_data['price']
    else:
        df['price'] = commodity_data.iloc[:, 0]

    # Optional columns
    if 'Volume' in commodity_data.columns:
        df['volume'] = commodity_data['Volume']
    if 'High' in commodity_data.columns:
        df['high'] = commodity_data['High']
    if 'Low' in commodity_data.columns:
        df['low'] = commodity_data['Low']
    if 'Open' in commodity_data.columns:
        df['open'] = commodity_data['Open']

    if usd_data is not None and 'Close' in usd_data.columns:
        usd_aligned = usd_data['Close'].reindex(df.index, method='ffill')
        df['usd_index'] = usd_aligned
    else:
        df['usd_index'] = 100

    df['returns'] = df['price'].pct_change()
    df = df.dropna()

    print(f"Prepared dataset with {len(df)} records for {commodity}")
    return df

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
        df.index = df.index.tz_localize(None)
        print(f"Data loaded from {filename}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Quick smoke test
    print("Testing data fetching (smoke test)...")
    gp = get_current_price('GC=F')
    sp = get_current_price('SI=F')
    print(f"Gold current (fallback): {gp}")
    print(f"Silver current (fallback): {sp}")
