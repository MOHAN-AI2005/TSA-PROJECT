import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')
LOG_FILE = os.path.join(BASE_DIR, 'audit_results.txt')

def perform_audit():
    with open(LOG_FILE, 'w') as f:
        f.write("="*60 + "\n")
        f.write("      DATASET INTEGRITY AUDIT\n")
        f.write("="*60 + "\n")
        
        if not os.path.exists(DATA_PATH):
            f.write(f"Error: Dataset not found at {DATA_PATH}\n")
            return

        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        
        # 1. Null Check
        nulls = df.isnull().sum()
        f.write("\n[1] NULL VALUE REPORT:\n")
        if nulls.sum() == 0:
            f.write("  SUCCESS: No null values found across 100% of the feature space.\n")
        else:
            f.write(str(nulls[nulls > 0]) + "\n")

        # 2. Outlier Analysis (Load)
        f.write("\n[2] OUTLIER SCAN (LOAD):\n")
        min_load = df['load'].min()
        max_load = df['load'].max()
        f.write(f"  Range: {min_load:,.0f} MW to {max_load:,.0f} MW\n")
        
        invalid_load = df[(df['load'] < 100000) | (df['load'] > 350000)]
        if len(invalid_load) == 0:
            f.write("  SUCCESS: All load values are within physical operational bounds.\n")
        else:
            f.write(f"  WARNING: Found {len(invalid_load)} suspicious load values.\n")

        # 3. Continuity Check
        f.write("\n[3] TEMPORAL CONTINUITY:\n")
        df = df.sort_values('date')
        date_diffs = df['date'].diff().dt.days
        gaps = date_diffs[date_diffs > 1]
        if len(gaps) == 0:
            f.write("  SUCCESS: Time series is continuous with no date gaps.\n")
        else:
            f.write(f"  WARNING: Found {len(gaps)} potential gaps in the sequence.\n")

        # 4. Feature Variance (Check for constant features)
        f.write("\n[4] FEATURE VARIANCE (CONSTANTS):\n")
        low_var = df.std(numeric_only=True)
        constant_cols = low_var[low_var == 0].index.tolist()
        if not constant_cols:
            f.write("  SUCCESS: All features exhibit valid statistical variance.\n")
        else:
            f.write(f"  WARNING: Constant features found: {constant_cols}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("      AUDIT COMPLETED\n")
        f.write("="*60 + "\n")

if __name__ == "__main__":
    perform_audit()
    print("Audit results written to audit_results.txt")
