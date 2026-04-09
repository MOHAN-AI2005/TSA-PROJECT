"""
extract_psp_data.py
====================
Extracts daily peak electricity demand (in MW) from NLDC PSP PDF reports
for all available years (2022, 2023, 2024, 2025).

INPUT  : data/raw/psp_reports/  (all *_NLDC_PSP.pdf files)
OUTPUT : data/processed/load_data_raw_all.csv
         (date, load, year — unsorted raw extraction)

Extraction logic:
  - Date is parsed from filename: DD.MM.YY_NLDC_PSP.pdf
  - Opens page 2 (index 1) of each PDF
  - Finds the "Maximum Demand Met During the Day" row
  - Takes the LAST number in that row (= TOTAL column in MW)
  - Falls back to scanning the next 5 lines if not found on same line
"""

import os
import re
import sys
import pdfplumber
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER  = os.path.join(BASE_DIR, "data", "raw", "psp_reports")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "load_data_raw_all.csv")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ---------------------------------------------------------------------------
# COLLECT ALL PDFS
# ---------------------------------------------------------------------------
all_pdfs = sorted([f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")])
print(f"Found {len(all_pdfs)} PDF files in: {PDF_FOLDER}")
print()

# ---------------------------------------------------------------------------
# EXTRACTION LOOP
# ---------------------------------------------------------------------------
records        = []
failed_files   = []
skipped_files  = []

for filename in tqdm(all_pdfs, desc="Extracting PDFs", unit="file"):
    file_path = os.path.join(PDF_FOLDER, filename)

    try:
        # Parse date from filename: "01.01.22_NLDC_PSP.pdf" -> 2022-01-01
        date_str = filename.split("_")[0]          # "01.01.22"
        date     = pd.to_datetime(date_str, format="%d.%m.%y")

        demand = None

        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) < 2:
                skipped_files.append((filename, "less than 2 pages"))
                continue

            text = pdf.pages[1].extract_text()

            if not text:
                skipped_files.append((filename, "page 2 empty"))
                continue

            lines = text.split("\n")

            # --- Primary: look for demand on same line ---
            for line in lines:
                if "Maximum Demand Met During the Day" in line:
                    numbers = re.findall(r"\d+", line)
                    if len(numbers) >= 6:
                        demand = int(numbers[-1])   # TOTAL column (last value)
                        break

            # --- Fallback: check next few lines after the header ---
            if demand is None:
                for i, line in enumerate(lines):
                    if "Maximum Demand Met During the Day" in line:
                        for j in range(i, min(i + 6, len(lines))):
                            nums = re.findall(r"\d+", lines[j])
                            if len(nums) >= 6:
                                demand = int(nums[-1])
                                break
                        if demand is not None:
                            break

        if demand is not None:
            records.append({
                "date": date,
                "load": demand,
                "year": date.year,
            })
        else:
            skipped_files.append((filename, "demand not found"))

    except Exception as e:
        failed_files.append((filename, str(e)[:80]))

# ---------------------------------------------------------------------------
# BUILD DATAFRAME & SAVE
# ---------------------------------------------------------------------------
df = pd.DataFrame(records)
df = df.sort_values("date").reset_index(drop=True)
df.to_csv(OUTPUT_FILE, index=False)

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("  EXTRACTION COMPLETE")
print("=" * 60)
print(f"  Total PDFs processed : {len(all_pdfs)}")
print(f"  Records extracted    : {len(df)}")
print(f"  Skipped (no data)    : {len(skipped_files)}")
print(f"  Failed (errors)      : {len(failed_files)}")
print(f"  Output               : {OUTPUT_FILE}")
print()

# Per-year breakdown
print("  Records per year:")
for yr, grp in df.groupby("year"):
    print(f"    {yr} : {len(grp)} records  "
          f"(range: {grp['date'].min().date()} to {grp['date'].max().date()})")

if skipped_files:
    print(f"\n  Skipped files (first 10):")
    for f, reason in skipped_files[:10]:
        print(f"    {f}  <- {reason}")

if failed_files:
    print(f"\n  Failed files:")
    for f, reason in failed_files:
        print(f"    {f}  <- {reason}")

print("=" * 60)