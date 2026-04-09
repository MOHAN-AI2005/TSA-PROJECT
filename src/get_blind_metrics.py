import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "ml__gradient_boosting_no_weather.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features_no_weather.pkl")

def get_blind_metrics():
    df = pd.read_csv(DATA_PATH)
    features = joblib.load(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    
    # Simple split for metric evaluation
    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size:]
    
    X_test = test_df[features]
    y_test = test_df['load']
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
    print(json.dumps(metrics))

if __name__ == "__main__":
    get_blind_metrics()
