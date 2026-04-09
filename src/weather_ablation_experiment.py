import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def run_experiment():
    print("Starting Weather Ablation Experiment...")
    
    # Load data
    data_path = "../data/processed/load_weather_all_features.csv"
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    
    # Drop lag features so we strictly compare "Calendar vs Weather+Calendar"
    # Otherwise the auto-regressive lag dominates both models identically
    features_all = [c for c in df.columns if c not in ['date', 'load', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']]
    weather_features = ['temp_max', 'temp_min', 'precipitation', 'wind_speed', 'temp_avg', 'temp_range']
    features_no_weather = [c for c in features_all if c not in weather_features]
    
    from sklearn.model_selection import train_test_split
    
    # Random split ensures both winter and extreme summer peaks are in the test set
    X_all = df[features_all]
    X_no_weather = df[features_no_weather]
    y = df['load']
    
    X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
    X_train_no, X_test_no, _, _ = train_test_split(X_no_weather, y, test_size=0.2, random_state=42)
    
    print("Training Model 1: Gradient Boosting (ALL FEATURES)...")
    gbm_all = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    gbm_all.fit(X_train_all, y_train)
    pred_all = gbm_all.predict(X_test_all)
    
    mae_all = mean_absolute_error(y_test, pred_all)
    rmse_all = np.sqrt(mean_squared_error(y_test, pred_all))
    
    print("Training Model 2: Gradient Boosting (NO WEATHER FEATURES)...")
    gbm_no = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    gbm_no.fit(X_train_no, y_train)
    pred_no = gbm_no.predict(X_test_no)
    
    mae_no = mean_absolute_error(y_test, pred_no)
    rmse_no = np.sqrt(mean_squared_error(y_test, pred_no))
    
    improvement = ((mae_no - mae_all) / mae_no) * 100
    
    print("\n==================================")
    print("       ABLATION RESULTS")
    print("==================================")
    print(f"With Weather MAE: {mae_all:,.2f} MW")
    print(f"Without Weather MAE: {mae_no:,.2f} MW")
    print(f"Improvement: {improvement:.2f}%\n")
    
    # Generate Chart
    out_dir = "../data/eda_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    metrics_data = pd.DataFrame({
        'Model': ['Gradient Boosting (With Weather)', 'Gradient Boosting (With Weather)', 
                  'Gradient Boosting (Weather Blind)', 'Gradient Boosting (Weather Blind)'],
        'Metric': ['MAE', 'RMSE', 'MAE', 'RMSE'],
        'Error_MW': [mae_all, rmse_all, mae_no, rmse_no]
    })
    
    ax = sns.barplot(data=metrics_data, x='Metric', y='Error_MW', hue='Model', palette=['#10b981', '#f43f5e'])
    plt.title('Weather Impact Ablation Study (Lower Error is Better)', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Error Margin (Megawatts)', fontsize=12)
    plt.xlabel('Evaluation Metric', fontsize=12)
    
    # Add exact text labels to the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f MW', padding=3, fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, '6_weather_ablation_study.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to '{os.path.abspath(plot_path)}'")

if __name__ == "__main__":
    run_experiment()
