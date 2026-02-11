"""
Download US stock data using yfinance.
Downloads top 50 stocks to complement existing data.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Try to import yfinance
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system("pip install yfinance -q")
    import yfinance as yf

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "us"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Top 50 US stocks (S&P 500 large caps)
US_STOCKS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "NFLX",
    "ADBE", "CRM", "ORCL", "CSCO", "IBM", "AVGO", "QCOM", "TXN", "NOW", "SNOW",

    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "PYPL", "AXP", "BLK",

    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",

    # Consumer
    "WMT", "HD", "COST", "NKE", "SBUX", "MCD", "DIS", "PG", "KO", "PEP",
]

def download_stock(symbol, start_date="2015-01-01", end_date=None):
    """Download a single stock's daily data."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df is None or len(df) == 0:
            print(f"  {symbol}: No data")
            return None

        # Keep only needed columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index.name = 'Date'

        return df

    except Exception as e:
        print(f"  {symbol}: Error - {str(e)[:50]}")
        return None

def main():
    # Check existing files
    existing = set(f.stem for f in OUTPUT_DIR.glob("*.parquet"))
    to_download = [s for s in US_STOCKS if s not in existing]

    print(f"Existing: {len(existing)} stocks")
    print(f"To download: {len(to_download)} stocks")
    print("-" * 50)

    if not to_download:
        print("All stocks already downloaded!")
        return

    success = 0
    failed = []

    for i, symbol in enumerate(to_download):
        print(f"[{i+1}/{len(to_download)}] Downloading {symbol}...", end=" ")

        df = download_stock(symbol)

        if df is not None and len(df) > 100:
            output_path = OUTPUT_DIR / f"{symbol}.parquet"
            df.to_parquet(output_path)
            print(f"OK ({len(df)} days)")
            success += 1
        else:
            failed.append(symbol)
            print("SKIP")

    print("-" * 50)
    print(f"Downloaded: {success}/{len(to_download)}")
    if failed:
        print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
