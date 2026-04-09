import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.pkl')

def train_weather_blind_model():
    print("Training Weather-Blind Model for Dashboard Toggle...")
    df = pd.read_csv(DATA_PATH)
    all_features = joblib.load(FEATURES_PATH)
    
    weather_features = ['temp_max', 'temp_min', 'temp_avg', 'temp_range', 'precipitation', 'wind_speed']
    blind_features = [f for f in all_features if f not in weather_features]
    
    X = df[blind_features]
    y = df['load']
    
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    
    model_name = "ml__gradient_boosting_no_weather.pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_name))
    
    # Save the feature list for this specific model too
    joblib.dump(blind_features, os.path.join(MODEL_DIR, "model_features_no_weather.pkl"))
    
    print(f"Success! Weather-blind model saved to {model_name}")

if __name__ == "__main__":
    train_weather_blind_model()
