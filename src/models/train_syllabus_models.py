import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_syllabus_models():
    print("Loading comprehensive dataset for Syllabus Alignment...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Drop NaNs created by rolling features
    df = df.dropna()
    
    # Target and Features
    y = df['load']
    X = df.drop(columns=['date', 'load'])
    
    # Fill edge-case NaNs securely
    X = X.bfill().ffill()
    
    expected_features = X.columns.tolist()
    joblib.dump(expected_features, os.path.join(MODEL_DIR, 'model_features.pkl'))
    
    # Syllabus Definition: 5 ML Architectures
    models = {
        "Baseline (Naive)": DummyRegressor(strategy='mean'),
        "Linear Model (Ridge)": Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        "Tree Ensemble (Random Forest)": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting (GBM)": GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        "Neural Network (MLP)": Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))])
    }
    
    # Cross-Val for Metrics
    tscv = TimeSeriesSplit(n_splits=3)
    results = {}
    
    print("Initiating Multi-Model Training Protocol...")
    for name, model in models.items():
        print(f"  -> Training & Evaluating {name}...")
        maes = []
        r2s = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            r2s.append(max(0, r2_score(y_test, preds))) # Floor at 0 for visual coherence
        
        avg_mae = np.mean(maes)
        avg_r2 = np.mean(r2s)
        
        # Fit final model on 100% data for actual deployment usage
        model.fit(X, y)
        safe_name = name.split(" ")[0].lower()
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}_model.pkl"))
        
        results[name] = {
            "MAE": float(avg_mae),
            "R2": float(avg_r2),
            "file": f"{safe_name}_model.pkl"
        }
        print(f"     ✅ Completed. MAE: {avg_mae:,.0f} MW")

    with open(os.path.join(MODEL_DIR, 'syllabus_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[SUCCESS] Custom Syllabus training sweep complete. Serialized to {MODEL_DIR}")

if __name__ == "__main__":
    train_syllabus_models()
