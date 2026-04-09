"""
download_psp_reports.py
========================
Downloads daily PSP (Power System Position) reports from NLDC (grid-india.in)
for multiple years. Skips files already downloaded.

SOURCE : https://grid-india.in/en/reports/daily-psp-report
FORMAT : DD.MM.YY_NLDC_PSP.pdf
OUTPUT : data/raw/psp_reports/

DATA NEEDS ANALYSIS:
  Year  | Days | Expected PDFs | Purpose
  ------|------|---------------|----------------------------------------
  2022  |  365 | ~301 (done)   | Already collected (82% coverage)
  2023  |  365 | to collect    | Year 1 of expansion
  2024  |  366 | to collect    | Year 2 (leap year)
  2025  |  365 | to collect    | Year 3 (latest complete year)
  TOTAL | ~1461| ~1100 expected| (some days may be missing on website)

WHY 3+ YEARS?
  - Time series models (SARIMAX, LSTM) need long history to learn seasonality
  - With only 2022 (~300 rows after cleaning), train/test split gives ~240/60
  - With 2022-2025 (~1100+ rows), train/test gives ~880/220 (much better!)
  - More data = better lag features, rolling stats, and seasonal patterns

HOW TO USE:
  python src/download_psp_reports.py
  (downloads 2023, 2024, 2025 by default — skips already-downloaded files)
"""

import os
import sys
import time
import requests
import urllib3
from datetime import datetime, timedelta

# Silence SSL warnings (grid-india.in has certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Project root = parent of src/
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_FOLDER = os.path.join(BASE_DIR, "data", "raw", "psp_reports")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# NLDC daily PSP report base URL
BASE_URL = "https://grid-india.in/en/reports/daily-psp-report"

# Browser-like headers to avoid 403 blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://grid-india.in/en/reports/daily-psp-report",
}

# ---------------------------------------------------------------------------
# YEARS TO DOWNLOAD
# ---------------------------------------------------------------------------
# 2022 is already collected (301/365 files exist)
# We now collect 2023, 2024, 2025

YEAR_RANGES = [
    (datetime(2022, 1, 1), datetime(2022, 12, 31)),
    (datetime(2023, 1, 1), datetime(2023, 12, 31)),
    (datetime(2024, 1, 1), datetime(2024, 12, 31)),
    (datetime(2025, 1, 1), datetime(2025, 12, 31)),
]

# ---------------------------------------------------------------------------
# URL GENERATION
# ---------------------------------------------------------------------------

def get_report_url(date_obj):
    """
    Returns the probable URL for a given date based on Grid-India's 
    changing folder structures for different financial years.
    """
    date_str = date_obj.strftime("%d.%m.%y")
    filename = f"{date_str}_NLDC_PSP.pdf"
    
    year = date_obj.year
    month = date_obj.month
    
    # Financial Year string (April to March)
    if month >= 4:
        fy_str = f"{year}-{year+1}"
    else:
        fy_str = f"{year-1}-{year}"
        
    # Pattern 1: Structured paths for FY 23-24 and 24-25
    if fy_str in ["2023-2024", "2024-2025"]:
        return f"https://webcdn.grid-india.in/files/grdw/uploads/daily-reports/psp-reports/{fy_str}/{filename}"
    
    # Pattern 2: Download manager for FY 22-23
    elif fy_str == "2022-2023":
        return f"https://webcdn.grid-india.in/files/grdw/uploads/download-manager-files/{filename}"
    
    # Pattern 3: Date-based path for recent data (2025-26)
    elif fy_str == "2025-2026":
        return f"https://webcdn.grid-india.in/files/grdw/{year}/{month:02d}/{filename}"
    
    # Fallback to base
    return f"https://grid-india.in/en/reports/daily-psp-report/{filename}"


# ---------------------------------------------------------------------------
# REQUEST SETTINGS
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS = 15      # wait up to 15s per request
DELAY_BETWEEN   = 0.5     # polite delay between requests (seconds)
MAX_RETRIES     = 3       # retry failed requests up to 3 times

# ---------------------------------------------------------------------------
# DOWNLOAD FUNCTION
# ---------------------------------------------------------------------------

def download_pdf(url, file_path, filename):
    """
    Attempt to download a single PDF. Returns 'downloaded', 'missing', or 'error'.
    Retries up to MAX_RETRIES times on connection errors.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                verify=False,
                timeout=TIMEOUT_SECONDS,
                stream=True,
            )

            if response.status_code == 200:
                # Verify it's actually a PDF (not an HTML error page)
                content_type = response.headers.get("Content-Type", "")
                content = response.content

                if len(content) < 1000:
                    # Suspiciously small — likely an error page, not a PDF
                    return "missing", f"too small ({len(content)} bytes)"

                with open(file_path, "wb") as f:
                    f.write(content)
                return "downloaded", f"{len(content):,} bytes"

            elif response.status_code == 404:
                return "missing", "404 Not Found"

            else:
                return "missing", f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return "error", f"Timeout after {MAX_RETRIES} retries"

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return "error", f"Connection error: {str(e)[:60]}"

        except Exception as e:
            return "error", str(e)[:80]

    return "error", "Max retries exceeded"


# ---------------------------------------------------------------------------
# MAIN DOWNLOAD LOOP
# ---------------------------------------------------------------------------

def run_download():
    print("=" * 65)
    print("  NLDC PSP REPORT DOWNLOADER")
    print("=" * 65)
    print(f"  Save folder : {SAVE_FOLDER}")
    print(f"  Years       : 2022, 2023, 2024, 2025")
    print()

    total_downloaded = 0
    total_skipped    = 0
    total_missing    = 0
    total_errors     = 0
    total_attempted  = 0

    for year_start, year_end in YEAR_RANGES:
        year = year_start.year
        year_downloaded = 0
        year_skipped    = 0
        year_missing    = 0
        year_errors     = 0

        current = year_start
        total_days = (year_end - year_start).days + 1

        print(f"--- Year {year} ({total_days} days) " + "-" * 35)

        while current <= year_end:
            date_str = current.strftime("%d.%m.%y")   # e.g. "01.01.23"
            filename  = f"{date_str}_NLDC_PSP.pdf"
            file_path = os.path.join(SAVE_FOLDER, filename)
            url       = get_report_url(current)
            total_attempted += 1

            # Skip if already downloaded
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                year_skipped    += 1
                total_skipped   += 1
                current         += timedelta(days=1)
                continue

            # Progress indicator
            day_num = (current - year_start).days + 1
            sys.stdout.write(
                f"\r  [{year}] Day {day_num:3d}/{total_days} | "
                f"Downloaded: {year_downloaded:3d} | "
                f"Missing: {year_missing:3d} | "
                f"Errors: {year_errors:2d}  "
            )
            sys.stdout.flush()

            status, info = download_pdf(url, file_path, filename)

            if status == "downloaded":
                year_downloaded += 1
                total_downloaded += 1
                # Print on new line so we don't overwrite progress
                sys.stdout.write("\n")
                print(f"    [OK] {filename}  ({info})")
            elif status == "missing":
                year_missing  += 1
                total_missing += 1
            elif status == "error":
                year_errors  += 1
                total_errors += 1
                sys.stdout.write("\n")
                print(f"    [ERR] {filename}  -> {info}")

            time.sleep(DELAY_BETWEEN)
            current += timedelta(days=1)

        # End-of-year summary
        sys.stdout.write("\n")
        print(f"\n  Year {year} Summary:")
        print(f"    Downloaded : {year_downloaded}")
        print(f"    Skipped    : {year_skipped}  (already existed)")
        print(f"    Missing    : {year_missing}  (not on website)")
        print(f"    Errors     : {year_errors}")
        print()

    # Grand total
    print("=" * 65)
    print("  DOWNLOAD COMPLETE")
    print("=" * 65)
    print(f"  Total Downloaded  : {total_downloaded}")
    print(f"  Total Skipped     : {total_skipped}  (already existed)")
    print(f"  Total Missing     : {total_missing}  (not available on website)")
    print(f"  Total Errors      : {total_errors}")
    print(f"  Attempted         : {total_attempted}")
    print()

    # Count all PDFs in folder
    all_pdfs = [f for f in os.listdir(SAVE_FOLDER) if f.endswith(".pdf")]
    print(f"  Total PDFs in folder: {len(all_pdfs)}")
    print(f"  Folder: {SAVE_FOLDER}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_download()