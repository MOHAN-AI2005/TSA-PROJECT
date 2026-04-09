"""
preprocessing.py
=================
Cleans and preprocesses raw extracted load data for all years (2022-2025).

INPUT  : data/processed/load_data_raw_all.csv
         (raw extraction, may have missing days, duplicates, bad values)

OUTPUT : data/processed/load_data_all_final.csv
         (clean, continuous, one row per day, interpolated)

Steps:
  1. Remove outliers: load <= 100,000 MW (clearly bad extractions)
  2. Remove duplicate dates (keep first)
  3. Sort by date
  4. Reindex to full 2022-01-01 to 2025-12-31 (1461 days)
  5. Linear interpolation to fill any missing days
  6. Forward-fill and back-fill for any remaining edge NaNs
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(BASE_DIR, "data", "processed", "load_data_raw_all.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "load_data_all_final.csv")

# ---------------------------------------------------------------------------
# DATE RANGE: full 4-year span
# ---------------------------------------------------------------------------
START_DATE = "2022-01-01"
END_DATE   = "2025-12-31"

# ---------------------------------------------------------------------------
# CLEAN
# ---------------------------------------------------------------------------

def clean_load_data():
    print("Loading raw extracted data ...")
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])

    print(f"  Raw records      : {len(df)}")
    print(f"  Date range (raw) : {df['date'].min().date()} to {df['date'].max().date()}")

    # --- Step 1: Remove bad extractions ---
    before = len(df)
    df = df[df['load'] > 100000]
    print(f"  After filter (>100k MW) : {len(df)}  (removed {before - len(df)} bad rows)")

    # --- Step 2: Remove duplicates ---
    df = df.drop_duplicates(subset='date', keep='first')
    print(f"  After deduplicate       : {len(df)}")

    # --- Step 3: Sort ---
    df = df.sort_values('date').set_index('date')

    # Keep only the 'load' column for the clean output
    df = df[['load']]

    # --- Step 4: Reindex to full date range ---
    actual_end_date = df.index.max().strftime('%Y-%m-%d')
    full_range = pd.date_range(start=START_DATE, end=actual_end_date, freq='D')
    missing_before = len(full_range) - len(df)
    df = df.reindex(full_range)
    df.index.name = 'date'
    print(f"  Missing days (before interpolation) : {missing_before}")

    # --- Step 5: Interpolate ---
    df['load'] = df['load'].interpolate(method='linear')

    # --- Step 6: Edge fill (first/last if still NaN) ---
    df['load'] = df['load'].ffill().bfill()

    remaining_nan = df['load'].isna().sum()
    if remaining_nan > 0:
        print(f"  WARNING: {remaining_nan} NaN values remain after interpolation!")
    else:
        print(f"  All missing days interpolated successfully.")

    # --- Save ---
    df.to_csv(OUTPUT_FILE)
    print(f"\n  Saved to: {OUTPUT_FILE}")

    # --- Per-year stats ---
    print()
    print("  Per-year summary:")
    print(f"  {'Year':<6} {'Rows':<6} {'Min Load':>10} {'Max Load':>10} {'Mean Load':>10}")
    print("  " + "-" * 46)
    for year in range(2022, 2026):
        subset = df[df.index.year == year]
        print(f"  {year:<6} {len(subset):<6} "
              f"{subset['load'].min():>10,.0f} "
              f"{subset['load'].max():>10,.0f} "
              f"{subset['load'].mean():>10,.0f}")

    print()
    print(f"  Total rows : {len(df)}")
    print(f"  Columns    : {list(df.reset_index().columns)}")
    print()
    print("  Preprocessing complete.")

    return df

if __name__ == "__main__":
    clean_load_data()