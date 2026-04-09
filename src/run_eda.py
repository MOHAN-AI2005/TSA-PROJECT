import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Starting Exploratory Data Analysis (EDA)...")
    
    # 1. Create output directory for figures
    out_dir = "../data/eda_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Load dataset
    data_path = "../data/processed/load_weather_all_features.csv"
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}")
        return
        
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Setup styling for professional plots
    sns.set_theme(style="whitegrid")
    
    # --- 1. Time Series Visualization ---
    print("Generating Time Series Plot...")
    plt.figure(figsize=(16, 6))
    plt.plot(df['date'], df['load'], color='#3498db', alpha=0.7, linewidth=1.5, label='Daily Max Demand')
    
    # Add a 30-day moving average trend line
    df_sorted = df.sort_values("date")
    plt.plot(df_sorted['date'], df_sorted['load'].rolling(window=30).mean(), color='#e74c3c', linewidth=2.5, label='30-Day Moving Average Trend')
    
    plt.title('Electricity Load Over Time (April 2022 - March 2025)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Load Demand (MW)', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_time_series_load.png'), dpi=300)
    plt.close()
    
    # --- 2. Monthly Trend ---
    print("Generating Monthly Trend Plot...")
    plt.figure(figsize=(12, 6))
    monthly_avg = df.groupby('month')['load'].mean().reset_index()
    
    # Map numbers to names for better readability
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    monthly_avg['month_name'] = monthly_avg['month'].map(month_names)
    
    sns.barplot(data=monthly_avg, x='month_name', y='load', palette='viridis', hue='month_name', legend=False)
    plt.title('Average Load by Month (Seasonal Pattern)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Load (MW)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_monthly_trend.png'), dpi=300)
    plt.close()
    
    # --- 3. Weekly Pattern (Weekend Dip) ---
    print("Generating Weekly Pattern Boxplot...")
    plt.figure(figsize=(12, 6))
    
    # 0=Monday, 6=Sunday
    day_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    df['day_name'] = df['day_of_week'].map(day_names)
    
    # Order for plotting
    order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.boxplot(data=df, x='day_name', y='load', order=order, palette='coolwarm', hue='day_name', legend=False)
    plt.title('Weekly Load Pattern: Industrial Weekday vs Weekend Dip', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Load (MW)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_weekly_pattern_boxplot.png'), dpi=300)
    plt.close()
    
    # --- 4. Weather Impact (Scatter Plots) ---
    print("Generating Weather Impact Scatter Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Max Temp vs Load
    sns.scatterplot(data=df, x='temp_max', y='load', alpha=0.6, ax=axes[0], color='#e67e22', edgecolor='w', s=60)
    axes[0].set_title('Load Demand vs Maximum Temperature', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Max Temperature (°C)', fontsize=12)
    axes[0].set_ylabel('Load (MW)', fontsize=12)
    
    # Precip vs Load
    sns.scatterplot(data=df, x='precipitation', y='load', alpha=0.6, ax=axes[1], color='#2980b9', edgecolor='w', s=60)
    axes[1].set_title('Load Demand vs Precipitation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Precipitation (mm)', fontsize=12)
    axes[1].set_ylabel('Load (MW)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_weather_impact_scatter.png'), dpi=300)
    plt.close()
    
    # --- 5. Correlation Heatmap ---
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(16, 12))
    
    # Drop non-numeric and intermediate encoded columns for cleaner heatmap
    cols_to_drop = ['month_sin', 'month_cos', 'day_sin', 'day_cos']
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Compute correlation
    corr = numeric_df.corr()
    
    # Create mask for upper triangle matrix to look cleaner
    import numpy as np
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=0.5, vmin=-1, vmax=1, 
                cbar_kws={"shrink": .8}, annot_kws={'size': 9})
    plt.title('Feature Correlation Heatmap (Pearson)', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '5_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"EDA successfully completed! 5 high-resolution plot files have been saved to '{os.path.abspath(out_dir)}'.")

if __name__ == "__main__":
    run_eda()
