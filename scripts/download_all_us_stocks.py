"""
Download comprehensive US common stock data via yfinance.

Targets ~4000-6000 US common stocks from NYSE + NASDAQ + AMEX.
Uses NASDAQ Trader registry for authoritative ticker list.

Usage:
    # Download all US common stocks (full universe)
    python scripts/download_all_us_stocks.py --all

    # Download S&P 500 only
    python scripts/download_all_us_stocks.py --sp500

    # Resume interrupted download
    python scripts/download_all_us_stocks.py --all --resume

    # Force re-download everything (uniform yfinance source)
    python scripts/download_all_us_stocks.py --all --force

    # Limit for testing
    python scripts/download_all_us_stocks.py --all --limit 50

    # Show database info
    python scripts/download_all_us_stocks.py --info

    # Refresh ticker universe from NASDAQ
    python scripts/download_all_us_stocks.py --refresh-universe

Time estimate: ~5000 stocks * ~1s/stock = ~1.5 hours
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stock_universe import USStockUniverse
from core.yfinance_fetcher import YFinanceFetcher
from core.tradingview_fetcher import USStockDatabase, DEFAULT_US_STOCKS


def load_download_state(path: Path) -> dict:
    """Load download progress state."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": {}, "skipped": []}


def save_download_state(path: Path, state: dict):
    """Save download progress state."""
    state["last_updated"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def apply_quality_filter(df, min_history: int, min_price: float, min_volume: float) -> bool:
    """Check if downloaded data passes quality filters."""
    if df is None or len(df) < min_history:
        return False
    if min_price > 0 and df["Close"].iloc[-1] < min_price:
        return False
    if min_volume > 0 and df["Volume"].mean() < min_volume:
        return False
    return True


def show_info(data_dir: Path):
    """Show database statistics."""
    db = USStockDatabase(data_dir=data_dir)

    print("=" * 70)
    print("US Stock Database Info")
    print("=" * 70)
    print(f"Data directory: {data_dir}")

    stocks = db.list_stocks()
    print(f"Stocks in database: {len(stocks)}")

    if db.metadata:
        summary = db.get_database_summary()
        if not summary.empty:
            print(f"\nTotal rows: {summary['rows'].sum():,}")
            print(f"Average rows/stock: {summary['rows'].mean():.0f}")
            print(f"Date range: {summary['start_date'].min()} to {summary['end_date'].max()}")

            # Source distribution
            if "source" in summary.columns:
                print(f"\nData sources:")
                for src, cnt in summary["source"].value_counts().items():
                    print(f"  {src}: {cnt}")

    # Check download state
    state_file = data_dir / "_download_state.json"
    if state_file.exists():
        state = load_download_state(state_file)
        print(f"\nDownload state:")
        print(f"  Completed: {len(state.get('completed', []))}")
        print(f"  Failed:    {len(state.get('failed', {}))}")
        print(f"  Skipped:   {len(state.get('skipped', []))}")
        if state.get("last_updated"):
            print(f"  Last updated: {state['last_updated']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download all US common stocks via yfinance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Download all US common stocks (~5000)")
    mode.add_argument("--sp500", action="store_true", help="Download S&P 500 only (~503)")
    mode.add_argument("--symbols", nargs="+", help="Download specific symbols")
    mode.add_argument("--info", action="store_true", help="Show database info")
    mode.add_argument("--refresh-universe", action="store_true", help="Refresh ticker list from NASDAQ")

    # Data params
    parser.add_argument("--start-date", default="2000-01-01", help="Start date (default: 2000-01-01)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory")

    # Quality filters
    parser.add_argument("--min-history", type=int, default=252,
                        help="Min trading days (default: 252 = ~1 year)")
    parser.add_argument("--min-price", type=float, default=1.0,
                        help="Min last close price (default: $1.00)")
    parser.add_argument("--min-volume", type=float, default=10000,
                        help="Min average daily volume (default: 10000)")

    # Execution
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded stocks")
    parser.add_argument("--force", action="store_true", help="Force re-download all")
    parser.add_argument("--delay", type=float, default=0.5, help="Base delay between requests (s)")
    parser.add_argument("--limit", type=int, help="Limit to first N stocks (for testing)")

    args = parser.parse_args()

    # Default data directory
    if args.data_dir is None:
        args.data_dir = Path(__file__).parent.parent / "data" / "raw" / "us"

    # Info mode
    if args.info:
        show_info(args.data_dir)
        return

    # Refresh universe
    if args.refresh_universe:
        print("Refreshing US stock universe from NASDAQ...")
        universe = USStockUniverse()
        stocks = universe.get_common_stocks()
        print(f"Found {len(stocks)} common stocks")
        print(f"\nExchange distribution:")
        print(stocks["exchange"].value_counts().to_string())
        print(f"\nSaved to: {Path(universe.cache_dir) / 'us_stock_universe.csv'}")
        return

    # Determine symbols
    if args.all:
        print("Fetching US stock universe from NASDAQ...")
        universe = USStockUniverse()
        symbols = universe.get_ticker_list("all")
        print(f"Universe: {len(symbols)} common stocks")
    elif args.sp500:
        symbols = list(DEFAULT_US_STOCKS)
        print(f"Using S&P 500: {len(symbols)} stocks")
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        parser.print_help()
        return

    if args.limit:
        symbols = symbols[:args.limit]

    # Initialize
    db = USStockDatabase(data_dir=args.data_dir)
    fetcher = YFinanceFetcher(min_delay=args.delay)
    state_file = args.data_dir / "_download_state.json"
    state = load_download_state(state_file)

    # Resume mode: skip already completed
    if args.resume and not args.force:
        completed_set = set(state.get("completed", []))
        existing_set = set(db.list_stocks())
        skip_set = completed_set | existing_set
        before = len(symbols)
        symbols = [s for s in symbols if s not in skip_set]
        print(f"Resume mode: skipping {before - len(symbols)} already downloaded, {len(symbols)} remaining")

    if not symbols:
        print("No stocks to download!")
        return

    # Print header
    print("=" * 70)
    print("US Stock Data Downloader (yfinance)")
    print("=" * 70)
    print(f"Target: {len(symbols)} stocks")
    print(f"Start date: {args.start_date}")
    print(f"Quality filters: history>={args.min_history}d, price>=${args.min_price}, vol>={args.min_volume}")
    print(f"Delay: {args.delay}s (adaptive)")
    print(f"Resume: {args.resume}, Force: {args.force}")
    print("=" * 70)

    # Download loop
    success = []
    failed = []
    skipped = []
    checkpoint_interval = 50

    pbar = tqdm(symbols, desc="Downloading", unit="stock")
    start_time = time.time()

    for i, symbol in enumerate(pbar):
        pbar.set_postfix({
            "OK": len(success),
            "Skip": len(skipped),
            "Fail": len(failed),
        })

        # Download
        df = fetcher.fetch_ohlcv(symbol, start_date=args.start_date)

        if df is not None and len(df) > 0:
            # Quality filter
            if apply_quality_filter(df, args.min_history, args.min_price, args.min_volume):
                try:
                    db.save_stock(symbol, df)
                    # Update metadata with source info
                    if symbol.upper() in db.metadata:
                        db.metadata[symbol.upper()]["source"] = "yfinance"
                    success.append(symbol)
                    state.setdefault("completed", []).append(symbol)
                except Exception as e:
                    tqdm.write(f"  [SAVE ERROR] {symbol}: {str(e)[:60]}")
                    failed.append(symbol)
                    state.setdefault("failed", {})[symbol] = str(e)[:100]
            else:
                skipped.append(symbol)
                state.setdefault("skipped", []).append(symbol)
        else:
            failed.append(symbol)
            state.setdefault("failed", {})[symbol] = "No data"

        # Periodic checkpoint
        if (i + 1) % checkpoint_interval == 0:
            save_download_state(state_file, state)
            db._save_metadata()

    # Final save
    save_download_state(state_file, state)
    db._save_metadata()

    elapsed = time.time() - start_time
    total = len(success) + len(failed) + len(skipped)

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Success:  {len(success):5d} / {total}")
    print(f"Skipped:  {len(skipped):5d} (quality filter)")
    print(f"Failed:   {len(failed):5d}")
    print(f"Time:     {elapsed/60:.1f} minutes ({elapsed/total:.1f}s/stock)")

    if failed:
        print(f"\nFailed symbols ({len(failed)}):")
        # Show first 30
        display = failed[:30]
        print("  " + " ".join(display))
        if len(failed) > 30:
            print(f"  ... and {len(failed) - 30} more")

        failed_file = args.data_dir / "failed_symbols.txt"
        with open(failed_file, "w") as f:
            f.write("\n".join(failed))
        print(f"  Saved to: {failed_file}")

    # Database status
    all_stocks = db.list_stocks()
    print(f"\nDatabase now contains: {len(all_stocks)} stocks")
    print("=" * 70)


if __name__ == "__main__":
    main()
