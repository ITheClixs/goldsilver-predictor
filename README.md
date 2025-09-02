# Introducing Gold & Silver Price Predictor 
## Screenshot from 26.08.2025
![Gold & Silver Predictor UI](images/Screenshot%202025-08-26%20at%2001.06.52.png)
## Screenshot from 02.09.2025
![Gold & Silver PredictorUI](images/Screenshot%202025-09-02%20at%2023.35.40.png)
## Overview
This project is a Flask-based web application that predicts the future prices of gold and silver (per gram in USD) for the next 1–30 days.  
It uses machine learning models trained on historical price data and technical indicators to forecast price movements.  
Additionally, the app provides trading signals (Strong Buy, Buy, Hold, Sell, Strong Sell) for each prediction horizon.

The app runs locally on `localhost` when executed, making it suitable for personal use and experimentation.

---

## Features
- Predict gold and silver prices for 1–30 days ahead.
- Display forecasted price for a specific day (e.g., "12 days later").
- Generate trading signals (buy/sell/hold).
- Fetch real-time gold and silver data from Yahoo Finance.
- Visualize predictions with plots.
- Simple web interface powered by Flask.

---

## Tech Stack
- **Backend**: Flask (Python web framework).
- **Machine Learning**: PyTorch, XGBoost, scikit-learn.
- **Data Handling**: Pandas, NumPy, joblib.
- **Data Source**: Yahoo Finance (via `yfinance`).
- **Visualization**: Matplotlib, Plotly.
- **HTTP Requests**: Requests library.

---

## Dependencies
The project uses the following Python libraries:
flask==2.3.3
pandas==2.0.3
numpy==1.24.3
torch==2.0.1
scikit-learn==1.3.0
xgboost==1.7.6
yfinance==0.2.21
requests==2.31.0
joblib==1.3.2
matplotlib==3.7.2
plotly==5.15.0

Make sure you are using **Python 3.9 or later**.

---

 ## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/gold-silver-prediction.git
   cd gold-silver-prediction

2. **Set up a virtual environment (recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt

   Running the Application
   
Train or load the pretrained models (scripts provided in models/).
Start the Flask server:
python app.py
Open your browser and go to:
http://127.0.0.1:5000/
Usage
Select prediction horizon (1–30 days).
 ## The app will display:
 - Predicted gold/silver price in USD per gram.
 - Suggested trading signal (Strong Buy, Buy, Hold, Sell, Strong Sell).
 - Visualization of price trends.

 ## Data Sources
Gold price (ticker: GC=F) and Silver price (ticker: SI=F) from Yahoo Finance.
Data is automatically fetched and updated using the yfinance library.

 ## Trading Signal Logic
Signals are generated based on predicted percentage change:
Strong Buy: Expected increase > 2% within horizon
Buy: Expected increase between 0.5% – 2%
Hold: Change between -0.5% and +0.5%
Sell: Expected decrease between -0.5% – -2%
Strong Sell: Expected decrease > -2%

