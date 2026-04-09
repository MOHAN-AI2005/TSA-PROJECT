import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def run_ml_models():
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')

    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Note: date shouldn't be a feature in tree models (can lead to overfitting on trend if not careful, 
    # though sometimes ordinal dates are used. We have cyclical calendar features so we will drop 'date')
    
    # Target: load
    y = df['load']
    X = df.drop(columns=['date', 'load'])

    features = X.columns.tolist()

    # ---------------------------------------------------------
    # 2. MODELS SETUP
    # ---------------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
    }

    # ---------------------------------------------------------
    # 3. TIME SERIES CROSS VALIDATION & EVALUATION
    # ---------------------------------------------------------
    print("\nStarting Time Series Cross Validation (5 splits)...")
    tss = TimeSeriesSplit(n_splits=5)

    results = {}

    for name, model in models.items():
        mae_scores = []
        rmse_scores = []
        
        for train_idx, test_idx in tss.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simple imputation if any NaNs slipped through (usually none if data is clean)
            X_train = X_train.fillna(method='bfill').fillna(method='ffill')
            X_test = X_test.fillna(method='bfill').fillna(method='ffill')
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mae_scores.append(mean_absolute_error(y_test, preds))
            rmse_scores.append(root_mean_squared_error(y_test, preds))
            
        results[name] = {
            'MAE': np.mean(mae_scores),
            'RMSE': np.mean(rmse_scores)
        }

    # ---------------------------------------------------------
    # 4. RESULTS SUMMARY
    # ---------------------------------------------------------
    print("\n===========================================================")
    print("  PHASE 4 MACHINE LEARNING MODELLING RESULTS (5-Fold TSCV)")
    print("===========================================================")
    print(f"{'Model Name':<25} | {'CV MAE':<10} | {'CV RMSE':<10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['MAE']:,.0f}      | {metrics['RMSE']:,.0f}")
    print("===========================================================")
    
    # ---------------------------------------------------------
    # 5. FEATURE IMPORTANCE (XGBoost)
    # ---------------------------------------------------------
    print("\nExtracting Feature Importances using XGBoost trained on full dataset...")
    # Train on full data for feature importance
    final_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    final_xgb.fit(X, y)
    
    importances = final_xgb.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Features (XGBoost):")
    print(feat_imp.head(10).to_string(index=False))

    # Optional: Plot feature importance
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='Importance', y='Feature', data=feat_imp)
    # plt.title('XGBoost Feature Importance')
    # plt.tight_layout()
    # plt.savefig(os.path.join(BASE_DIR, 'dashboard', 'feature_importance.png'))

if __name__ == "__main__":
    run_ml_models()
