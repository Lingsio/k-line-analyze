"""Debug dataset loading to see which stocks are loaded"""
import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
data_dir = PROJECT_ROOT / 'data' / 'raw' / 'us'

# List all parquet files
all_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
print(f"Total parquet files found: {len(all_files)}")
print(f"Files: {sorted(all_files)[:10]}...")

# Try loading each file
loaded = []
failed = []

for file in sorted(all_files):
    ticker = file.replace('.parquet', '')
    try:
        file_path = os.path.join(data_dir, file)
        df = pd.read_parquet(file_path)

        # Check required columns
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in req_cols):
            failed.append((ticker, f"Missing columns: {set(req_cols) - set(df.columns)}"))
            continue

        # Check data size
        if len(df) < 25:  # window_size (20) + prediction_horizon (5)
            failed.append((ticker, f"Insufficient data: {len(df)} rows"))
            continue

        loaded.append(ticker)

    except Exception as e:
        failed.append((ticker, str(e)))

print(f"\nSuccessfully loadable: {len(loaded)} stocks")
print(f"Failed to load: {len(failed)} stocks")

if failed:
    print("\nFailed stocks:")
    for ticker, reason in failed[:20]:
        print(f"  {ticker}: {reason}")

print(f"\nLoaded stocks: {loaded}")
