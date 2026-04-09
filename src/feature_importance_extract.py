import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def extract_importance():
    print("Extracting Feature Importance from Champion Model...")
    
    model_path = "../models/ml__gradient_boosting_model.pkl"
    features_path = "../models/model_features.pkl"
    out_dir = "../data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "feature_importance.png")
    
    # Load Model and Feature Names
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print(f"Error: Could not find model files at {model_path} or {features_path}")
        return
        
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # Extract importances
    importances = model.feature_importances_
    
    # Create DataFrame and get top 10
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    top_10 = importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Features requested to be explicitly highlighted
    highlight_features = ['lag_1', 'lag_7', 'temp_max', 'month_sin', 'is_weekend']
    
    # Assign specific colors based on the highlight requirement
    def get_color(feature):
        if feature in highlight_features:
            return '#e74c3c'  # Vibrant Red/Orange for highlighted features
        return '#34495e'      # Sleek Slate for standard features
        
    colors = [get_color(f) for f in top_10['Feature']]
    
    # Initialize Plot
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", rc={"axes.facecolor":"#f8f9fa"})
    
    # Plot horizontal bar chart
    ax = sns.barplot(data=top_10, x='Importance', y='Feature', palette=colors)
    
    plt.title('Top 10 Feature Importances (Champion Gradient Boosting Model)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Relative Feature Importance (Gini Importance)', fontsize=14, fontweight='bold')
    plt.ylabel('Feature Name', fontsize=14, fontweight='bold')
    
    # Style axes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=13, fontweight='bold')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=5, fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Add a custom legend to explain the colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Targeted Analytical Feature'),
        Patch(facecolor='#34495e', label='Standard Contributing Feature')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n==============================================")
    print(f"Top 5 Identified Drivers:")
    print(top_10.head(5).to_string(index=False))
    print(f"==============================================")
    print(f"\nSuccess! Feature Importance plot saved to '{os.path.abspath(out_path)}'")


if __name__ == "__main__":
    extract_importance()
