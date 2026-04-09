import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOAD_CLEAN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_data_all_final.csv')
WEATHER_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'weather_data.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')

def run_feature_pipeline():
    print("="*60)
    print("  TSA FEATURE ENGINEERING PIPELINE (2022-2025)")
    print("="*60)

    # 1. LOAD DATA
    print(f"Loading clean load data: {LOAD_CLEAN_PATH}")
    load_df = pd.read_csv(LOAD_CLEAN_PATH)
    load_df['date'] = pd.to_datetime(load_df['date'])

    print(f"Loading weather data: {WEATHER_RAW_PATH}")
    weather_df = pd.read_csv(WEATHER_RAW_PATH)
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # 2. MERGE
    print("Merging Load and Weather data ...")
    df = pd.merge(load_df, weather_df, on='date', how='inner')
    print(f"  Merged Shape: {df.shape}")

    # 3. CALENDAR FEATURES
    print("Generating Calendar Features ...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # Cyclical Encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 4. LAG FEATURES (Target-based)
    print("Generating Lag Features ...")
    df['lag_1'] = df['load'].shift(1)
    df['lag_7'] = df['load'].shift(7)
    df['lag_30'] = df['load'].shift(30)

    # 5. ROLLING FEATURES
    print("Generating Rolling Features ...")
    df['rolling_mean_7'] = df['load'].rolling(window=7).mean()
    df['rolling_std_7'] = df['load'].rolling(window=7).std()

    # 6. CLEAN UP
    print("Cleaning up ...")
    initial_len = len(df)
    df = df.dropna()
    print(f"  Rows dropped (due to lags/rolling): {initial_len - len(df)}")

    # 7. SAVE
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSUCCESS: Feature-engineered dataset saved at {OUTPUT_PATH}")
    print(f"  Final columns: {df.columns.tolist()}")
    print(f"  Data range: {df['date'].min().date()} to {df['date'].max().date()}")
    print("="*60)

if __name__ == "__main__":
    run_feature_pipeline()
