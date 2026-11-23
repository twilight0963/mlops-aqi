"""Load the registered champion model and predict next-day PM2.5."""

import mlflow.pyfunc
import pandas as pd

from src.config import load_params

def load_champion():
    params = load_params()
    model_name = params["mlflow"]["registered_model_name"]
    model_uri = f"models:/{model_name}@champion"

    print(f"Loading model: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def predict(model, input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return float(prediction[0])


if __name__ == "__main__":
    model = load_champion()

    sample_input = {
        "pm25_today": 120.0,
        "wind_kmph": 8.0,
        "humidity_pct": 65.0,
        "temp_c": 24.0,
        "traffic_index": 7.0,
        "stubble_fires": 2.0,
        "day_type": "weekday",
        "station_zone": "Central",
    }

    result = predict(model, sample_input)

    print("\nInput:")
    print(pd.DataFrame([sample_input]).to_string(index=False))

    print(f"\nPredicted PM2.5 for next day: {result:.2f}")
