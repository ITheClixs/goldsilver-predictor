#!/usr/bin/env python3
"""
Setup script for Gold & Silver Price Forecasting Web App
Run this script to set up the environment and test the application
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True

def create_directory_structure():
    """Create necessary directories"""
    print("\nCreating directory structure...")
    
    directories = [
        'data',
        'models', 
        'templates',
        'static'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    # Install packages
    if platform.system() == "Windows":
        command = "pip install -r requirements.txt"
    else:
        command = "pip3 install -r requirements.txt"
    
    return run_command(command, "Installing Python packages")

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting module imports...")
    
    required_modules = [
        'flask',
        'pandas', 
        'numpy',
        'torch',
        'sklearn',
        'xgboost',
        'yfinance',
        'requests',
        'joblib'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages manually")
        return False
    
    print("✓ All modules imported successfully")
    return True

def test_data_fetching():
    """Test data fetching functionality"""
    print("\nTesting data fetching...")
    
    try:
        # Test the data fetching module
        sys.path.append(os.getcwd())
        from data.fetch_data import get_current_price
        
        print("Testing gold price fetch...")
        gold_price = get_current_price('GC=F')
        if gold_price != "N/A":
            print(f"✓ Gold price fetched: ${gold_price}")
        else:
            print("⚠️  Gold price fetch returned N/A (may be due to market hours)")
        
        print("Testing silver price fetch...")
        silver_price = get_current_price('SI=F')
        if silver_price != "N/A":
            print(f"✓ Silver price fetched: ${silver_price}")
        else:
            print("⚠️  Silver price fetch returned N/A (may be due to market hours)")
        
        return True
        
    except Exception as e:
        print(f"✗ Data fetching test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing if live data is not available"""
    print("\nCreating sample data for testing...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample gold data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
        gold_prices = 1800 + np.cumsum(np.random.randn(1000) * 2)
        
        gold_data = pd.DataFrame({
            'price': gold_prices,
            'volume': np.random.randint(10000, 50000, 1000),
            'high': gold_prices * 1.01,
            'low': gold_prices * 0.99,
            'open': gold_prices * (1 + np.random.randn(1000) * 0.001),
            'returns': pd.Series(gold_prices).pct_change(),
            'usd_index': 100 + np.cumsum(np.random.randn(1000) * 0.1)
        }, index=dates)
        
        gold_data.to_csv('data/gold_sample.csv')
        print("✓ Sample gold data created")
        
        # Generate sample silver data
        silver_prices = 25 + np.cumsum(np.random.randn(1000) * 0.1)
        
        silver_data = pd.DataFrame({
            'price': silver_prices,
            'volume': np.random.randint(5000, 25000, 1000),
            'high': silver_prices * 1.01,
            'low': silver_prices * 0.99,
            'open': silver_prices * (1 + np.random.randn(1000) * 0.001),
            'returns': pd.Series(silver_prices).pct_change(),
            'usd_index': 100 + np.cumsum(np.random.randn(1000) * 0.1)
        }, index=dates)
        
        silver_data.to_csv('data/silver_sample.csv')
        print("✓ Sample silver data created")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
        return False

def run_quick_test():
    """Run a quick test of the core functionality"""
    print("\nRunning quick functionality test...")
    
    try:
        sys.path.append(os.getcwd())
        
        # Test technical indicators
        from data.indicators import calculate_indicators
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'price': 1800 + np.cumsum(np.random.randn(100) * 2),
            'volume': np.random.randint(1000, 10000, 100),
            'high': np.random.uniform(1800, 1900, 100),
            'low': np.random.uniform(1700, 1800, 100),
            'open': np.random.uniform(1750, 1850, 100),
            'returns': np.random.randn(100) * 0.01,
            'usd_index': 100 + np.random.randn(100) * 0.1
        }, index=dates)
        
        # Test indicators calculation
        df_with_indicators = calculate_indicators(test_data)
        if len(df_with_indicators) > 0:
            print("✓ Technical indicators calculation works")
        else:
            print("✗ Technical indicators calculation failed")
            return False
        
        # Test LSTM model creation (without training)
        from models.lstm_model import LSTMModel
        lstm_model = LSTMModel(sequence_length=10, num_epochs=1)
        print("✓ LSTM model can be created")
        
        # Test XGBoost model creation
        from models.xgb_model import XGBoostModel
        xgb_model = XGBoostModel()
        print("✓ XGBoost model can be created")
        
        # Test ensemble
        from models.ensemble import EnsemblePredictor
        ensemble = EnsemblePredictor()
        print("✓ Ensemble predictor can be created")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("Gold & Silver Price Forecasting Web App Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directory_structure():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Package installation failed. Please install manually using:")
        print("pip install -r requirements.txt")
        response = input("\nContinue setup anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test data fetching
    test_data_fetching()
    
    # Create sample data
    create_sample_data()
    
    # Run quick test
    if not run_quick_test():
        print("\n⚠️  Quick test failed, but setup will continue")
    
    print("\n" + "="*60)
    print("Setup completed!")
    print("="*60)
    
    print("\nTo start the application:")
    print("1. Run: python app.py")
    print("2. Open your browser and go to: http://localhost:5000")
    print("\nFirst-time usage:")
    print("- The first prediction may take 2-3 minutes as models are trained")
    print("- Subsequent predictions will be much faster")
    print("- Models are saved and will be reused")
    
    print("\nTroubleshooting:")
    print("- If data fetching fails, check your internet connection")
    print("- If training is slow, reduce num_epochs in LSTM model")
    print("- For any issues, check the console output for error messages")
    
    print("\nDisclaimer: This is for educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()