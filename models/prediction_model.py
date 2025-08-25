import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForTimeSeriesForecasting
from data.fetch_data import prepare_dataset

class PredictionModel:
    def __init__(self, model_name="huggingface/time-series-transformer-tourism-monthly"):
        self.model_name = model_name
        self.model = AutoModelForTimeSeriesForecasting.from_pretrained(self.model_name)
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
            
            # The time-series transformer model works best on the raw price series.
            time_series = df['price'].tolist()
            
            # The model expects the data as a tensor of past values.
            context_length = self.model.config.context_length
            past_values = torch.tensor(time_series[-context_length:]).unsqueeze(0)

            # Generate the forecast
            forecast = self.model.generate(
                past_values=past_values,
                prediction_length=horizon,
                num_beams=5,
                early_stopping=True
            )
            
            # The output is a sequence of predicted values.
            # We will take the prediction for the requested horizon.
            predicted_price = forecast[0, -1].item()

            return predicted_price

        except Exception as e:
            print(f"Prediction failed: {e}")
            return None