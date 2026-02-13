"""
Download all S&P 500 stock data from TradingView.

This script ONLY downloads data, no image processing.
Use this first, then run build_ohlc_library.py to create images.

Usage:
    # Download all 503 stocks
    python scripts/download_sp500_data.py --all --delay 2
    
    # Download first 50 stocks (test)
    python scripts/download_sp500_data.py --limit 50
    
    # Resume interrupted download
    python scripts/download_sp500_data.py --all --resume
    
    # Check what we have
    python scripts/download_sp500_data.py --info
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tradingview_fetcher import TradingViewFetcher, USStockDatabase, DEFAULT_US_STOCKS

# Optional yfinance fallback
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def download_yfinance(symbol, start_date):
    """Download data using yfinance as fallback."""
    if not HAS_YFINANCE:
        return None
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, auto_adjust=True)
        if df is None or df.empty:
            return None
        # Standardize columns
        df = df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low',
            'Close': 'Close', 'Volume': 'Volume',
        })
        # Keep only OHLCV columns
        cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[cols]
        # Remove timezone from index
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = 'Date'
        return df
    except Exception:
        return None


def download_with_retry(fetcher, symbol, start_date, max_retries=3, base_delay=2.0, use_yfinance=True):
    """Download data with retry logic. Falls back to yfinance on failure."""
    # Try TradingView first
    for attempt in range(max_retries):
        try:
            df = fetcher.fetch_history(symbol, start_date)
            if df is not None and len(df) > 100:
                return df, "tv"
            # If we got data but it's too short, don't retry
            if df is not None:
                break
        except Exception as e:
            error_msg = str(e).lower()
            if "timezone" in error_msg or "parse" in error_msg:
                break

            wait_time = base_delay * (2 ** attempt)
            if attempt < max_retries - 1:
                time.sleep(wait_time)

    # Fallback to yfinance
    if use_yfinance and HAS_YFINANCE:
        df = download_yfinance(symbol, start_date)
        if df is not None and len(df) > 100:
            return df, "yf"

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Download S&P 500 stock data")
    parser.add_argument("--all", action="store_true", help="Download all 503 stocks")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols")
    parser.add_argument("--limit", type=int, help="Limit to first N stocks")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/us"))
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded stocks")
    parser.add_argument("--info", action="store_true", help="Show database info and exit")
    parser.add_argument("--force", action="store_true", help="Force re-download existing stocks")
    parser.add_argument("--no-yfinance", action="store_true", help="Disable yfinance fallback")

    args = parser.parse_args()
    
    # Initialize database
    db = USStockDatabase(data_dir=args.data_dir)
    
    # Show info mode
    if args.info:
        print("=" * 70)
        print("S&P 500 Data Database Info")
        print("=" * 70)
        print(f"Data directory: {args.data_dir}")
        
        existing = db.list_stocks()
        print(f"Stocks in database: {len(existing)}")
        
        if existing and len(existing) > 0:
            df_meta = db.get_database_summary()
            if not df_meta.empty:
                print(f"\nTotal rows: {df_meta['rows'].sum():,}")
                print(f"Date range: {df_meta['start_date'].min()} to {df_meta['end_date'].max()}")
                
                print("\nMissing S&P 500 stocks:")
                missing = set(DEFAULT_US_STOCKS) - set(existing)
                if missing:
                    print(f"  {len(missing)} missing: {sorted(missing)[:20]}...")
                else:
                    print("  None! All 503 stocks present.")
        
        print("\nDatabase stocks:")
        for s in sorted(existing)[:30]:
            info = db.get_stock_info(s)
            if info:
                print(f"  {s}: {info['rows']} rows ({info['start_date']} to {info['end_date']})")
        if len(existing) > 30:
            print(f"  ... and {len(existing) - 30} more")
        
        print("=" * 70)
        return
    
    # Determine symbols to download
    if args.all:
        symbols = DEFAULT_US_STOCKS
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_US_STOCKS[:10]
    
    if args.limit:
        symbols = symbols[:args.limit]
    
    # Filter already downloaded
    if args.resume and not args.force:
        existing = set(db.list_stocks())
        symbols = [s for s in symbols if s not in existing]
        print(f"Resuming: {len(symbols)} stocks remaining")
    
    if not symbols:
        print("No stocks to download!")
        return
    
    print("=" * 70)
    print("S&P 500 Data Downloader")
    print("=" * 70)
    print(f"Target: {len(symbols)} stocks")
    print(f"Start date: {args.start_date}")
    print(f"Delay: {args.delay}s")
    print(f"Resume mode: {args.resume}")
    print(f"yfinance fallback: {'enabled' if HAS_YFINANCE and not args.no_yfinance else 'disabled'}")
    print("=" * 70)

    fetcher = TradingViewFetcher()
    use_yf = HAS_YFINANCE and not args.no_yfinance

    success_tv = []
    success_yf = []
    failed = []
    skipped = []

    pbar = tqdm(symbols, desc="Downloading", unit="stock")

    for symbol in pbar:
        pbar.set_postfix({
            "OK": len(success_tv) + len(success_yf),
            "Fail": len(failed),
            "Skip": len(skipped),
        })

        # Check if exists and recent
        if not args.force:
            existing = db.load_stock(symbol)
            if existing is not None and len(existing) > 100:
                try:
                    last_date = pd.Timestamp(existing.index[-1])
                    if last_date.tz is not None:
                        last_date = last_date.tz_localize(None)
                    days_old = (pd.Timestamp.now() - last_date).days
                    if days_old < 7:
                        skipped.append(symbol)
                        continue
                except Exception:
                    pass

        # Download with retry + fallback
        df, source = download_with_retry(
            fetcher, symbol, args.start_date,
            base_delay=args.delay, use_yfinance=use_yf,
        )

        if df is not None and len(df) > 100:
            try:
                db.save_stock(symbol, df)
                if source == "yf":
                    success_yf.append(symbol)
                else:
                    success_tv.append(symbol)
            except Exception as e:
                tqdm.write(f"  [ERROR] {symbol} save failed: {str(e)[:60]}")
                failed.append(symbol)
        else:
            failed.append(symbol)

        # Delay between requests
        time.sleep(args.delay)

    # Summary
    total_success = len(success_tv) + len(success_yf)
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Success:  {total_success:3d} / {len(symbols)}")
    if success_tv:
        print(f"  via TradingView: {len(success_tv)}")
    if success_yf:
        print(f"  via yfinance:    {len(success_yf)}")
    print(f"Skipped:  {len(skipped):3d} (already have recent data)")
    print(f"Failed:   {len(failed):3d}")

    if failed:
        print(f"\nFailed symbols ({len(failed)}):")
        print("  " + " ".join(failed))

        # Save failed list for retry
        failed_file = args.data_dir / "failed_symbols.txt"
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed))
        print(f"\nSaved failed list to: {failed_file}")

    print("=" * 70)

    # Show current status
    existing = db.list_stocks()
    print(f"\nDatabase now contains: {len(existing)} / {len(DEFAULT_US_STOCKS)} S&P 500 stocks")


if __name__ == "__main__":
    main()
