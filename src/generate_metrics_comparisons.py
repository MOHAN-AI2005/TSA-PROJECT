import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_comparison_charts():
    print("Generating Comprehensive Model Comparison Chart...")
    
    metrics_path = "models/syllabus_duel_metrics.json"
    out_dir = "../data/eda_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "7_all_models_metrics_comparison.png")
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    # Extract data into lists
    models = []
    maes = []
    rmses = []
    mapes = []
    
    for model_name, info in metrics.items():
        # Shorten names for readability on x-axis
        short_name = model_name.replace('Classical - ', '').replace('ML - ', '')
        if short_name == "Gradient Boosting":
            short_name = "Gradient Boost (Best)"
            
        models.append(short_name)
        maes.append(info.get('MAE', 0))
        rmses.append(info.get('RMSE', 0))
        mapes.append(info.get('MAPE', 0))
        
    df = pd.DataFrame({
        'Model': models,
        'MAE': maes,
        'RMSE': rmses,
        'MAPE': mapes
    })
    
    # We will use subplots because MAPE is a percentage (scale 0-100) 
    # while MAE/RMSE are in Megawatts (scale 0-200,000)
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    sns.set_theme(style="whitegrid", rc={"axes.facecolor":"#f8f9fa"})
    
    # Helper to highlight the best model
    def get_colors(models_series):
        return ['#10b981' if 'Best' in m else '#34495e' for m in models_series]
        
    colors = get_colors(df['Model'])
    
    # 1. Plot MAE
    ax0 = sns.barplot(data=df, x='Model', y='MAE', palette=colors, ax=axes[0])
    axes[0].set_title('Mean Absolute Error (MAE) - Lower is Better', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_ylabel('MAE (MW)', fontsize=12, fontweight='bold')
    for container in ax0.containers:
        ax0.bar_label(container, fmt='%d', padding=3, fontsize=10)
        
    # 2. Plot RMSE
    ax1 = sns.barplot(data=df, x='Model', y='RMSE', palette=colors, ax=axes[1])
    axes[1].set_title('Root Mean Square Error (RMSE) - Lower is Better', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_ylabel('RMSE (MW)', fontsize=12, fontweight='bold')
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%d', padding=3, fontsize=10)
        
    # 3. Plot MAPE
    ax2 = sns.barplot(data=df, x='Model', y='MAPE', palette=colors, ax=axes[2])
    axes[2].set_title('Mean Absolute Percentage Error (MAPE %) - Lower is Better', fontsize=14, fontweight='bold', pad=10)
    axes[2].set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2f%%', padding=3, fontsize=10)
        
    # Formatting X-axis for all
    for ax in axes:
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=20, labelsize=11)
        
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#10b981', label='Champion Model (Gradient Boosting)'),
        Patch(facecolor='#34495e', label='Standard Models')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccess! 3-Panel Metrics Comparison chart saved to '{os.path.abspath(out_path)}'")


if __name__ == "__main__":
    generate_comparison_charts()
