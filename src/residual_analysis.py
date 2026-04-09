import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split

def run_residual_analysis():
    print("Starting Residual Analysis for Gradient Boosting Champion...")
    
    # 1. Setup paths
    model_path = "../models/ml__gradient_boosting_model.pkl"
    features_path = "../models/model_features.pkl"
    data_path = "../data/processed/load_weather_all_features.csv"
    out_dir = "../data/eda_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "9_residual_analysis.png")
    
    # 2. Load data and model
    if not all(os.path.exists(p) for p in [model_path, features_path, data_path]):
        print("Error: Missing model or data files.")
        return
        
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # Use the same split logic as earlier experiments for consistency (Random 80/20)
    X = df[feature_names]
    y = df['load']
    dates = df['date']
    
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.2, random_state=42
    )
    
    # 3. Compute Residuals
    predictions = model.predict(X_test)
    residuals = y_test - predictions
    
    # Metrics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    print(f"\n[RESIDUAL_METRICS]")
    print(f"Mean Residual (Bias): {mean_res:,.2f} MW")
    print(f"Residual Std Dev: {std_res:,.2f} MW")
    
    # 4. Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot A: Histogram (Distribution)
    sns.histplot(residuals, kde=True, ax=axes[0], color='#6366f1', bins=40)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Baseline')
    axes[0].axvline(mean_res, color='orange', linestyle='-', linewidth=2, label=f'Mean Bias: {mean_res:.1f} MW')
    axes[0].set_title('Distribution of Residuals (Actual - Predicted)', fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Residual Error (MW)', fontsize=12)
    axes[0].legend()
    
    # Plot B: Residual vs Time
    # Sort dates for a clean time series line
    res_ts = pd.DataFrame({'date': dates_test, 'residual': residuals}).sort_values('date')
    axes[1].scatter(res_ts['date'], res_ts['residual'], alpha=0.5, color='#10b981', s=15, label='Daily Residuals')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    
    # Add a rolling mean to see if bias changes over time
    axes[1].plot(res_ts['date'], res_ts['residual'].rolling(window=10).mean(), color='orange', linewidth=2, label='10-Day Bias Trend')
    
    axes[1].set_title('Residuals Over Time (Heteroscedasticity Check)', fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Residual Error (MW)', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"\nSuccess! Residual Analysis plots saved to '{os.path.abspath(out_path)}'")

if __name__ == "__main__":
    run_residual_analysis()
