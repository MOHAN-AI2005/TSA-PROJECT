import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os

def run_incremental_experiment():
    print("Starting Incremental Feature Group Experiment...")
    
    # Load data
    data_path = "../data/processed/load_weather_all_features.csv"
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    
    # Define groups
    lag_features = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']
    calendar_features = ['day_of_week', 'day_of_month', 'month', 'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    weather_features = ['temp_max', 'temp_min', 'temp_avg', 'precipitation', 'wind_speed']
    
    # Define experimental settings
    groups = {
        "1. Lags Only": lag_features,
        "2. Lags + Calendar": lag_features + calendar_features,
        "3. Lags + Calendar + Weather": lag_features + calendar_features + weather_features
    }
    
    y = df['load']
    results = []
    
    # Split
    # We use random split to ensure we have a representative sample of all seasons in the test set
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    
    for name, features in groups.items():
        print(f"Executing: {name} (Features: {len(features)})")
        X = df[features]
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]
        
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        results.append({"Group": name, "MAE": mae})
        print(f"   -> MAE: {mae:,.2f} MW")

    # Plotting
    out_dir = "../data/eda_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "8_incremental_feature_improvement.png")
    
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Custom color palette (fading from gray to blue to green)
    palette = ["#94a3b8", "#6366f1", "#10b981"]
    
    ax = sns.barplot(data=res_df, x='Group', y='MAE', palette=palette, hue='Group', legend=False)
    plt.title('Incremental Accuracy Gain by Feature Group (Lower is Better)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Mean Absolute Error (MAE) in MW', fontsize=12, fontweight='bold')
    plt.xlabel('Feature Inclusion Stage', fontsize=12, fontweight='bold')
    
    # Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f MW', padding=3, fontsize=11, fontweight='bold')
        
    # Annotate percentage improvements
    mae_1 = results[0]['MAE']
    mae_2 = results[1]['MAE']
    mae_3 = results[2]['MAE']
    
    imp1 = ((mae_1 - mae_2) / mae_1) * 100
    imp2 = ((mae_2 - mae_3) / mae_2) * 100
    total_imp = ((mae_1 - mae_3) / mae_1) * 100
    
    plt.text(0.5, (mae_1 + mae_2)/2, f"-{imp1:.1f}% Error", ha='center', fontweight='bold', color='#4f46e5')
    plt.text(1.5, (mae_2 + mae_3)/2, f"-{imp2:.1f}% Error", ha='center', fontweight='bold', color='#059669')
    
    plt.figtext(0.5, 0.01, f"Total Feature Engineering Gain: {total_imp:.1f}% Error Reduction from Baseline", 
                ha='center', fontsize=12, fontweight='bold', color='#1e293b')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"\nSuccess! Incremental improvement chart saved to '{os.path.abspath(out_path)}'")
    
    # Output results for log capture
    print("\n[SUMMARY_RESULTS]")
    for r in results:
        print(f"{r['Group']} | {r['MAE']}")

if __name__ == "__main__":
    run_incremental_experiment()
