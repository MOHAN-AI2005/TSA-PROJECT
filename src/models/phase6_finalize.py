import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def finalize_model():
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading complete 2022 dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. SEPARATE FEATURES AND TARGET
    # ---------------------------------------------------------
    y = df['load']
    X = df.drop(columns=['date', 'load'])
    
    # Fill any edge-case NaNs with forward/backward fills
    X = X.bfill().ffill()

    # ---------------------------------------------------------
    # 3. CONSOLIDATE RESULTS (HISTORICAL SUMMARY)
    # ---------------------------------------------------------
    print("\n===========================================================")
    print("      FINAL PHASE 6: MODEL COMPARISON SUMMARY")
    print("===========================================================")
    print(f"{'Algorithm Type':<30} | {'Model':<25} | {'Metric (Out-Of-Sample MAE)'}")
    print("-" * 80)
    print(f"{'Classical Stats (1-Split)':<30} | {'SARIMAX (Weather)':<25} | 71,274 MW")
    print(f"{'Classical Stats (1-Split)':<30} | {'Prophet (Base)':<25} | 27,055 MW")
    print(f"{'Deep Learning (1-Split)':<30} | {'PyTorch LSTM':<25} | 13,430 MW")
    print(f"{'Machine Learning (5-Fold CV)':<30} | {'Linear Regression':<25} | 7,987 MW")
    print(f"{'Machine Learning (5-Fold CV)':<30} | {'XGBoost':<25} | 4,774 MW")
    print(f"{'Machine Learning (5-Fold CV)':<30} | {'Random Forest':<25} | 4,168 MW   <-- [CHAMPION]")
    print("===========================================================\n")
    
    print("Selecting [Random Forest (Max Depth: 10, Estimators: 100)] as the final production model.")

    # ---------------------------------------------------------
    # 4. TRAIN FINAL MODEL ON 100% DATASET
    # ---------------------------------------------------------
    print("\nTraining Final Champion Model on the FULL dataset (335 items)...")
    final_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    final_model.fit(X, y)
    
    # Get internal R2 on full training data just to verify fit
    full_preds = final_model.predict(X)
    full_r2 = r2_score(y, full_preds)
    print(f"Full Dataset Fit Accuracy (R2): {full_r2:.4f}")

    # ---------------------------------------------------------
    # 5. SAVE MODEL TO DISK
    # ---------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, 'champion_rf_model.pkl')
    joblib.dump(final_model, model_path)
    
    # Also save the exact features list so the dashboard knows what coordinates it needs
    features_path = os.path.join(MODEL_DIR, 'model_features.pkl')
    joblib.dump(X.columns.tolist(), features_path)

    print(f"\n[SUCCESS] Final model safely serialized to: {model_path}")
    print(f"[SUCCESS] Expected features serialized to : {features_path}")

if __name__ == "__main__":
    finalize_model()
