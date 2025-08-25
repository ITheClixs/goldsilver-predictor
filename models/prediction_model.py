import pandas as pd
from transformers import pipeline
from data.fetch_data import prepare_dataset

class PredictionModel:
    def __init__(self, model="huggingface/time-series-transformer-tourism-monthly"):
        self.pipe = pipeline("time-series-forecasting", model=model)
        self.is_trained = False

    def train(self, commodity='gold', force_retrain=False):
        # The pre-trained model does not need to be trained.
        # In a real-world scenario, this model should be fine-tuned on the specific dataset.
        print("Using pre-trained model directly without fine-tuning.")
        self.is_trained = True
        return True

    def predict(self, commodity='gold', horizon=1):
        if not self.is_trained:
            self.train(commodity)

        try:
            # Prepare data
            df = prepare_dataset(commodity)
            if df is None:
                return None
            
            # The pipeline expects a pandas DataFrame with a DatetimeIndex.
            # The target column should be named 'target'.
            df = df.rename(columns={'price': 'target'})
            
            # The pipeline will make predictions for `prediction_length` steps.
            forecast = self.pipe(df, prediction_length=horizon)
            
            # The output is a list of dictionaries, each with a 'mean' value.
            # We will take the last prediction.
            predicted_price = forecast[-1]['mean']

            return predicted_price

        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
