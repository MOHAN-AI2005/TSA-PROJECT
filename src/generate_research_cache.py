import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHAMPION_MODEL = os.path.join(MODEL_DIR, "ml__gradient_boosting_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
CACHE_PATH = os.path.join(MODEL_DIR, "research_cache.json")

def generate_cache():
    print("Generating Research Cache (Importance + Residuals)...")
    
    # Check dependencies
    if not all(os.path.exists(p) for p in [DATA_PATH, CHAMPION_MODEL, FEATURES_PATH]):
        print("Error: Missing model or data files.")
        return
        
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(CHAMPION_MODEL)
    feature_names = joblib.load(FEATURES_PATH)
    
    # 1. Feature Importance
    importances = model.feature_importances_
    importance_list = [
        {"feature": name, "importance": float(imp)} 
        for name, imp in zip(feature_names, importances)
    ]
    # Sort and take top 10
    importance_list = sorted(importance_list, key=lambda x: x['importance'], reverse=True)[:10]
    
    # 2. Residuals
    # Use random split consistent with earlier experiments
    X = df[feature_names]
    y = df['load']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    predictions = model.predict(X_test)
    residuals = (y_test - predictions).values
    
    # Histogram of residuals
    counts, bin_edges = np.histogram(residuals, bins=30)
    bins = []
    for i in range(len(counts)):
        mid = (bin_edges[i] + bin_edges[i+1]) / 2
        bins.append({"bin": float(mid), "count": int(counts[i])})
        
    # Sample Residuals over time (last 100 points for UI performance)
    sample_indices = np.linspace(0, len(residuals)-1, 100, dtype=int)
    # We'll use the indices of the test set to get dates if needed, but for the scatter 
    # we just need index vs value.
    residual_scatter = [
        {"index": int(i), "value": float(residuals[i])} 
        for i in sample_indices
    ]
    
    cache = {
        "importance": importance_list,
        "residuals": {
            "distribution": bins,
            "scatter": residual_scatter,
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals))
        }
    }
    
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=4)
        
    print(f"Success! Research cache saved to {CACHE_PATH}")

if __name__ == "__main__":
    generate_cache()
