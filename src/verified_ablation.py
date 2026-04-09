import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import os

def run_verified_ablation():
    # Load data from project root context
    data_path = r"c:/Users/reddy/OneDrive/Documents/TSA-PROJECT/data/processed/load_weather_all_features.csv"
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    
    # Strictly isolate weather impact by dropping auto-regressive lags
    features_all = [c for c in df.columns if c not in ['date', 'load', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']]
    weather_features = ['temp_max', 'temp_min', 'precipitation', 'wind_speed', 'temp_avg', 'temp_range']
    features_no_weather = [c for c in features_all if c not in weather_features]
    
    y = df['load']
    X_train_all, X_test_all, y_train, y_test = train_test_split(df[features_all], y, test_size=0.2, random_state=42)
    X_train_no, X_test_no, _, _ = train_test_split(df[features_no_weather], y, test_size=0.2, random_state=42)
    
    params = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
    
    gbm_all = GradientBoostingRegressor(**params)
    gbm_all.fit(X_train_all, y_train)
    pred_all = gbm_all.predict(X_test_all)
    mae_all = mean_absolute_error(y_test, pred_all)
    
    gbm_no = GradientBoostingRegressor(**params)
    gbm_no.fit(X_train_no, y_train)
    pred_no = gbm_no.predict(X_test_no)
    mae_no = mean_absolute_error(y_test, pred_no)
    
    improvement = ((mae_no - mae_all) / mae_no) * 100
    
    print(f"VERIFIED_MAE_WITH_WEATHER: {mae_all}")
    print(f"VERIFIED_MAE_WITHOUT_WEATHER: {mae_no}")
    print(f"VERIFIED_IMPROVEMENT: {improvement}%")

if __name__ == "__main__":
    run_verified_ablation()
