import pandas as pd
import numpy as np

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: Price series
        window: RSI period (default 14)
    
    Returns:
        RSI values (0-100)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        Dictionary with MACD line, signal line, and histogram
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data: Price series
        window: Moving average period (default 20)
        num_std: Number of standard deviations (default 2)
    
    Returns:
        Dictionary with upper, middle, and lower bands
    """
    middle_band = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }

def calculate_volatility(data, window=20):
    """
    Calculate rolling volatility (standard deviation of returns)
    
    Args:
        data: Price series
        window: Rolling window period
    
    Returns:
        Rolling volatility
    """
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return volatility

def calculate_momentum(data, window=10):
    """Calculate price momentum"""
    return data / data.shift(window) - 1

def calculate_indicators(df):
    """
    Calculate all technical indicators for the dataset
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional indicator columns
    """
    print("Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
    df_indicators = df.copy()
    
    # Moving Averages
    df_indicators['sma_7'] = calculate_sma(df['price'], 7)
    df_indicators['sma_14'] = calculate_sma(df['price'], 14)
    df_indicators['sma_30'] = calculate_sma(df['price'], 30)
    df_indicators['sma_200'] = calculate_sma(df['price'], 200)
    
    df_indicators['ema_12'] = calculate_ema(df['price'], 12)
    df_indicators['ema_26'] = calculate_ema(df['price'], 26)
    
    # RSI
    df_indicators['rsi'] = calculate_rsi(df['price'])
    
    # MACD
    macd_dict = calculate_macd(df['price'])
    df_indicators['macd'] = macd_dict['macd']
    df_indicators['macd_signal'] = macd_dict['signal']
    df_indicators['macd_histogram'] = macd_dict['histogram']
    
    # Bollinger Bands
    bb_dict = calculate_bollinger_bands(df['price'])
    df_indicators['bb_upper'] = bb_dict['upper']
    df_indicators['bb_middle'] = bb_dict['middle']
    df_indicators['bb_lower'] = bb_dict['lower']
    df_indicators['bb_width'] = (bb_dict['upper'] - bb_dict['lower']) / bb_dict['middle']
    df_indicators['bb_position'] = (df['price'] - bb_dict['lower']) / (bb_dict['upper'] - bb_dict['lower'])
    
    # Volatility
    df_indicators['volatility_20'] = calculate_volatility(df['price'], 20)
    df_indicators['volatility_60'] = calculate_volatility(df['price'], 60)
    
    # Momentum indicators
    df_indicators['momentum_10'] = calculate_momentum(df['price'], 10)
    df_indicators['momentum_20'] = calculate_momentum(df['price'], 20)
    
    # Price relative to moving averages
    df_indicators['price_sma7_ratio'] = df['price'] / df_indicators['sma_7']
    df_indicators['price_sma30_ratio'] = df['price'] / df_indicators['sma_30']
    df_indicators['price_sma200_ratio'] = df['price'] / df_indicators['sma_200']
    
    # Volume indicators (if volume available)
    if 'volume' in df.columns:
        df_indicators['volume_sma_20'] = calculate_sma(df['volume'], 20)
        df_indicators['volume_ratio'] = df['volume'] / df_indicators['volume_sma_20']
    
    # USD Index indicators (if available)
    if 'usd_index' in df.columns:
        df_indicators['usd_sma_20'] = calculate_sma(df['usd_index'], 20)
        df_indicators['usd_momentum'] = calculate_momentum(df['usd_index'], 10)
        df_indicators['price_usd_ratio'] = df['price'] / df['usd_index']
    
    # Lag features (previous day values)
    for col in ['price', 'returns', 'rsi', 'macd']:
        if col in df_indicators.columns:
            df_indicators[f'{col}_lag1'] = df_indicators[col].shift(1)
            df_indicators[f'{col}_lag2'] = df_indicators[col].shift(2)
            df_indicators[f'{col}_lag3'] = df_indicators[col].shift(3)
    
    # Clean up NaN values
    df_indicators = df_indicators.dropna()
    
    print(f"Indicators calculated. Dataset shape: {df_indicators.shape}")
    print(f"Available columns: {list(df_indicators.columns)}")
    
    return df_indicators

def get_feature_columns():
    """
    Get list of feature columns for ML models
    
    Returns:
        List of column names to use as features
    """
    feature_cols = [
        'sma_7', 'sma_14', 'sma_30', 'sma_200',
        'ema_12', 'ema_26',
        'rsi',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_width', 'bb_position',
        'volatility_20', 'volatility_60',
        'momentum_10', 'momentum_20',
        'price_sma7_ratio', 'price_sma30_ratio', 'price_sma200_ratio',
        'returns_lag1', 'returns_lag2', 'returns_lag3',
        'rsi_lag1', 'macd_lag1'
    ]
    
    # Add optional features if available
    optional_features = [
        'volume_ratio',
        'usd_sma_20', 'usd_momentum', 'price_usd_ratio'
    ]
    
    return feature_cols, optional_features

if __name__ == "__main__":
    # Test indicator calculations
    print("Testing technical indicators...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    prices = 1000 + np.cumsum(np.random.randn(1000) * 0.02) * 50
    
    df_test = pd.DataFrame({
        'price': prices,
        'volume': np.random.randint(1000, 10000, 1000),
        'high': prices * 1.01,
        'low': prices * 0.99,
        'open': prices * (1 + np.random.randn(1000) * 0.001),
        'returns': pd.Series(prices).pct_change(),
        'usd_index': 100 + np.cumsum(np.random.randn(1000) * 0.001) * 10
    }, index=dates)
    
    # Calculate indicators
    df_with_indicators = calculate_indicators(df_test)
    
    print(f"Test dataset shape: {df_with_indicators.shape}")
    print("\nSample indicators:")
    print(df_with_indicators[['price', 'sma_30', 'rsi', 'macd', 'volatility_20']].tail())