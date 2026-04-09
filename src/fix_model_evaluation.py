import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
METRICS_PATH = os.path.join(MODEL_DIR, 'syllabus_duel_metrics.json')

def fix_leaderboard():
    print("🚀 Starting Unified Model Evaluation for Leaderboard Correction...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    
    # 2. Define Features
    # Target and drops
    y = df['load']
    drop_cols = ['date', 'load']
    
    # Full Feature Set (Champion)
    all_features = [c for c in df.columns if c not in drop_cols]
    
    # Weather-Blind Feature Set (Ablation)
    weather_keywords = ['temp', 'precipitation', 'wind']
    blind_features = [c for c in all_features if not any(k in c.lower() for k in weather_keywords)]
    
    print(f"Full Features: {len(all_features)}")
    print(f"Blind Features: {len(blind_features)}")
    
    # 3. Split Data (Chronological slice prevents future leakage in time-series)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train_full = train_df[all_features]
    X_test_full = test_df[all_features]
    X_train_blind = train_df[blind_features]
    X_test_blind = test_df[blind_features]
    y_train = train_df['load']
    y_test = test_df['load']
    
    params = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
    
    # 4. Train & Eval Champion (With Weather)
    print("\nTraining Champion Model (With Weather)...")
    gbm_full = GradientBoostingRegressor(**params)
    gbm_full.fit(X_train_full, y_train)
    pred_full = gbm_full.predict(X_test_full)
    
    mae_full = mean_absolute_error(y_test, pred_full)
    rmse_full = np.sqrt(mean_squared_error(y_test, pred_full))
    mape_full = np.mean(np.abs((y_test - pred_full) / y_test)) * 100
    
    # 5. Train & Eval Ablation (No Weather)
    print("Training Ablation Model (No Weather)...")
    gbm_blind = GradientBoostingRegressor(**params)
    gbm_blind.fit(X_train_blind, y_train)
    pred_blind = gbm_blind.predict(X_test_blind)
    
    mae_blind = mean_absolute_error(y_test, pred_blind)
    rmse_blind = np.sqrt(mean_squared_error(y_test, pred_blind))
    mape_blind = np.mean(np.abs((y_test - pred_blind) / y_test)) * 100
    
    # 6. Log Truth
    print("\n" + "="*40)
    print("       FINAL VERIFIED METRICS")
    print("="*40)
    print(f"WITH WEATHER:    MAE={mae_full:,.2f} | RMSE={rmse_full:,.2f} | MAPE={mape_full:.2f}%")
    print(f"WITHOUT WEATHER: MAE={mae_blind:,.2f} | RMSE={rmse_blind:,.2f} | MAPE={mape_blind:.2f}%")
    
    if mae_full < mae_blind:
        print("\n✅ Verification SUCCESS: 'With Weather' model outperforms 'No Weather'.")
    else:
        print("\n❌ Verification FAILED: Metrics are still illogical. Check feature leakage.")
    print("="*40)
    
    # 7. Update JSON safely
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    metrics["ML - Gradient Boosting"] = {
        "Category": "Machine Learning",
        "MAE": mae_full,
        "RMSE": rmse_full,
        "MAPE": mape_full,
        "Method": "ml",
        "file": "ml__gradient_boosting_model.pkl"
    }
    
    metrics["ML - Gradient Boosting (No Weather)"] = {
        "Category": "Machine Learning",
        "MAE": mae_blind,
        "RMSE": rmse_blind,
        "MAPE": mape_blind,
        "Method": "ml",
        "file": "ml__gradient_boosting_no_weather.pkl"
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nUpdated {METRICS_PATH}")
    
    # 8. Update Model Files
    joblib.dump(gbm_full, os.path.join(MODEL_DIR, "ml__gradient_boosting_model.pkl"))
    joblib.dump(gbm_blind, os.path.join(MODEL_DIR, "ml__gradient_boosting_no_weather.pkl"))
    joblib.dump(all_features, os.path.join(MODEL_DIR, "model_features.pkl"))
    joblib.dump(blind_features, os.path.join(MODEL_DIR, "model_features_no_weather.pkl"))
    print("Serialized updated models and feature lists.")

if __name__ == "__main__":
    fix_leaderboard()
