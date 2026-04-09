import os
import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore") # Suppress statsmodels convergence warnings during automation

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_syllabus_duel():
    print("Loading comprehensive dataset for Classical vs ML Duel...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna()
    
    y = df['load'].values
    X_df = df.drop(columns=['date', 'load'])
    X_df = X_df.bfill().ffill()
    X = X_df.values
    
    expected_features = X_df.columns.tolist()
    joblib.dump(expected_features, os.path.join(MODEL_DIR, 'model_features.pkl'))
    
    # Validation Split Strategy (Holdout of last 30 days for strict chronological tests)
    train_size = len(df) - 30
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    results = {}

    print("\n--- PHASE 1: CLASSICAL (UNIVARIATE TIME SERIES) ---")
    classical_methods = ["Naive Baseline", "Simple Exponential Smoothing", "Holt-Winters", "SARIMA"]
    
    # 1. Naive Baseline (Last observed carries forward constantly)
    baseline_preds = np.repeat(y_train[-1], len(y_test))
    mae = mean_absolute_error(y_test, baseline_preds)
    results["Classical - Naive"] = {"Category": "Classical", "MAE": float(mae), "Method": "naive"}
    print(f"Classical - Naive | MAE: {mae:,.0f} MW")
    
    # 2. Simple Exponential Smoothing
    try:
        ses_model = SimpleExpSmoothing(y_train).fit(smoothing_level=0.5, optimized=False)
        ses_preds = ses_model.forecast(len(y_test))
        mae = mean_absolute_error(y_test, ses_preds)
        results["Classical - SES"] = {"Category": "Classical", "MAE": float(mae), "Method": "ses"}
        print(f"Classical - SES   | MAE: {mae:,.0f} MW")
    except Exception as e: print(f"SES failed {e}")
    
    # 3. Holt-Winters
    try:
        hw_model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=7).fit()
        hw_preds = hw_model.forecast(len(y_test))
        mae = mean_absolute_error(y_test, hw_preds)
        results["Classical - Holt-Winters"] = {"Category": "Classical", "MAE": float(mae), "Method": "hw"}
        print(f"Classical - HW    | MAE: {mae:,.0f} MW")
    except Exception as e: print(f"HW failed {e}")
    
    # 4. SARIMA (Static params for speed)
    try:
        sarima_model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(0,1,1,7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sarima_preds = sarima_model.forecast(steps=len(y_test))
        mae = mean_absolute_error(y_test, sarima_preds)
        results["Classical - SARIMA"] = {"Category": "Classical", "MAE": float(mae), "Method": "sarima"}
        print(f"Classical - SARIMA| MAE: {mae:,.0f} MW")
    except Exception as e: print(f"SARIMA failed {e}")
    
    
    print("\n--- PHASE 2: MACHINE LEARNING (MULTIVARIATE SUPERVISED) ---")
    ml_models = {
        "ML - Ridge": Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        "ML - Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "ML - Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        "ML - SVR": Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=10, epsilon=0.1))]),
        "ML - Neural Net (MLP)": Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))])
    }
    
    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # Fit on 100% data to serialize for the UI Dashboard
        model.fit(X, y)
        safe_name = name.lower().replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").strip()
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}_model.pkl"))
        
        results[name] = {"Category": "Machine Learning", "MAE": float(mae), "Method": "ml", "file": f"{safe_name}_model.pkl"}
        print(f"{name:20s} | MAE: {mae:,.0f} MW")

    # Serialize metrics dictionary 
    with open(os.path.join(MODEL_DIR, 'syllabus_duel_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[SUCCESS] Classical vs ML duel complete. Saved metrics to {MODEL_DIR}")

if __name__ == "__main__":
    train_syllabus_duel()
