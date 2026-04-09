import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def run_rolling_validation():
    print("Starting Rolling Window Validation (Stability Check)...")
    
    # 1. Setup paths
    features_path = "../models/model_features.pkl"
    data_path = "../data/processed/load_weather_all_features.csv"
    out_dir = "../data/eda_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "10_rolling_validation_mae.png")
    
    # 2. Load data
    if not os.path.exists(data_path) or not os.path.exists(features_path):
        print("Error: Missing data or feature files.")
        return
        
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_names = joblib.load(features_path)
    
    # Parameters for sliding window
    initial_train_days = 365 # Start with 1 full year of training
    forecast_window = 30     # Test on next 30 days
    step_size = 30           # Move forward by 30 days each step
    
    total_days = len(df)
    results = []
    
    print(f"Dataset coverage: {total_days} days.")
    print("Beginning rolling iterations...")
    
    current_train_end = initial_train_days
    
    while current_train_end + forecast_window <= total_days:
        # Split data
        train_df = df.iloc[:current_train_end]
        test_df = df.iloc[current_train_end : current_train_end + forecast_window]
        
        X_train, y_train = train_df[feature_names], train_df['load']
        X_test, y_test = test_df[feature_names], test_df['load']
        
        # Train model (using champion GBM configuration)
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # Store result with the date representing the middle of the test window
        mid_date = test_df['date'].iloc[len(test_df)//2]
        results.append({
            'date': mid_date,
            'mae': mae,
            'train_size': len(train_df)
        })
        
        # Advance the window
        current_train_end += step_size
    
    if not results:
        print("Error: No validation windows could be created. Dataset might be too short.")
        return
        
    res_df = pd.DataFrame(results)
    avg_rolling_mae = res_df['mae'].mean()
    
    print(f"\n[ROLLING_METRICS]")
    print(f"Total Windows Evaluated: {len(res_df)}")
    print(f"Average Rolling MAE: {avg_rolling_mae:,.2f} MW")
    
    # 3. Plotting
    plt.figure(figsize=(15, 7))
    sns.set_theme(style="whitegrid")
    
    # Plot MAE line
    plt.plot(res_df['date'], res_df['mae'], color='#6366f1', marker='o', linewidth=2.5, markersize=8, label='30-Day Window MAE')
    
    # Add horizontal line for average
    plt.axhline(avg_rolling_mae, color='#f43f5e', linestyle='--', linewidth=2, label=f'Avg Rolling MAE: {avg_rolling_mae:,.0f} MW')
    
    # Highlighting seasons (Visual contextualization)
    plt.title('Rolling Window Validation (Stability Across Seasons)', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Mean Absolute Error (MW)', fontsize=14)
    plt.xlabel('Validation Window Date', fontsize=14)
    plt.legend(fontsize=12)
    
    # Fill background to denote years if possible
    # (Simple approach: just a clean plot for graduation defense)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"\nSuccess! Rolling Validation plot saved to '{os.path.abspath(out_path)}'")

if __name__ == "__main__":
    run_rolling_validation()
