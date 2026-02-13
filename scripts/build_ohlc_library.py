"""
Build OHLC Bar Image Library from local data.

Prerequisites:
    1. First run: python scripts/download_sp500_data.py --all
    2. Then run this script to build images

Usage:
    # Build from all existing data
    python scripts/build_ohlc_library.py --all --window 20 --stride 5
    
    # Build from specific stocks
    python scripts/build_ohlc_library.py --symbols AAPL MSFT GOOGL
    
    # Quick test (first 10 stocks)
    python scripts/build_ohlc_library.py --limit 10
    
    # Resume interrupted build
    python scripts/build_ohlc_library.py --all --resume
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tradingview_fetcher import USStockDatabase, DEFAULT_US_STOCKS
from core.utils.ohlc_renderer import ImageLibrary


def build_library(
    symbols: list,
    data_dir: Path,
    library_dir: Path,
    window_size: int = 20,
    stride: int = 5,
    include_volume: bool = False,
) -> dict:
    """
    Build OHLC image library from local stock data.
    """
    print("=" * 70)
    print("Building OHLC Image Library")
    print("=" * 70)
    print(f"Data source: {data_dir}")
    print(f"Output: {library_dir}")
    print(f"Window: {window_size} bars, Stride: {stride}")
    print(f"Include volume: {include_volume}")
    print("=" * 70)
    
    db = USStockDatabase(data_dir=data_dir)
    library = ImageLibrary(library_dir, n_bars=window_size, include_volume=include_volume)
    
    total_images = 0
    stats = []
    errors = []
    
    for symbol in tqdm(symbols, desc="Processing"):
        try:
            # Load data
            df = db.load_stock(symbol)
            if df is None:
                errors.append((symbol, "No data file"))
                continue
            
            if len(df) < window_size + 50:  # Need some buffer
                errors.append((symbol, f"Insufficient data: {len(df)} rows"))
                continue
            
            # Build images
            result = library.add_stock_data(
                symbol=symbol,
                df=df,
                stride=stride,
            )
            
            total_images += result['num_images']
            stats.append({
                'symbol': symbol,
                'num_images': result['num_images'],
                'data_rows': len(df),
            })
            
        except Exception as e:
            error_msg = str(e)[:80]
            errors.append((symbol, error_msg))
            tqdm.write(f"  Error {symbol}: {error_msg}")
    
    print(f"\nBuild complete: {total_images} images from {len(stats)} stocks")
    
    if errors:
        print(f"Errors: {len(errors)} stocks")
        # Save error log
        error_file = library_dir / "build_errors.txt"
        with open(error_file, 'w') as f:
            for symbol, error in errors:
                f.write(f"{symbol}: {error}\n")
        print(f"Error log: {error_file}")
    
    return {
        'total_images': total_images,
        'stats': stats,
        'errors': errors,
    }


def summarize(library_dir: Path):
    """Print library summary."""
    print("\n" + "=" * 70)
    print("Image Library Summary")
    print("=" * 70)
    
    library = ImageLibrary(library_dir, n_bars=20)
    metadata = library.get_metadata()
    
    if metadata.empty:
        print("Library is empty!")
        return
    
    print(f"Total images: {len(metadata):,}")
    print(f"Unique symbols: {metadata['symbol'].nunique()}")
    
    # Symbol stats
    symbol_counts = metadata['symbol'].value_counts()
    print(f"\nTop 10 symbols:")
    for symbol, count in symbol_counts.head(10).items():
        print(f"  {symbol}: {count:,} images")
    
    # Return distribution
    if 'return' in metadata.columns:
        returns = metadata['return']
        print(f"\nReturn distribution:")
        print(f"  Mean: {returns.mean():.4f}")
        print(f"  Std:  {returns.std():.4f}")
        print(f"  Up:   {(returns > 0).mean():.1%}")
        print(f"  Down: {(returns < 0).mean():.1%}")
    
    # Image dimensions
    sample_img = None
    for symbol_dir in (library_dir / "processed").iterdir():
        if symbol_dir.is_dir():
            imgs = list(symbol_dir.glob("*.png"))
            if imgs:
                from PIL import Image
                sample_img = Image.open(imgs[0])
                print(f"\nImage dimensions: {sample_img.size}")
                print(f"Image mode: {sample_img.mode}")
                break
    
    # Save summary
    summary_file = library_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"OHLC Image Library Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"\nTotal images: {len(metadata):,}\n")
        f.write(f"Unique symbols: {metadata['symbol'].nunique()}\n")
        if sample_img:
            f.write(f"Image size: {sample_img.size}\n")
        f.write(f"\nSymbol counts:\n")
        for symbol, count in symbol_counts.items():
            f.write(f"  {symbol}: {count}\n")
    
    print(f"\nSummary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Build OHLC image library")
    parser.add_argument("--all", action="store_true", help="Process all S&P 500 stocks with data")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols")
    parser.add_argument("--limit", type=int, help="Limit to first N stocks")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/us"))
    parser.add_argument("--library-dir", type=Path, default=Path("data/ohlc_library"))
    parser.add_argument("--window", type=int, default=20, choices=[5, 20, 60])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--volume", action="store_true", help="Include volume bars")
    parser.add_argument("--resume", action="store_true", help="Skip already processed symbols")
    
    args = parser.parse_args()
    
    db = USStockDatabase(data_dir=args.data_dir)
    
    # Determine symbols
    if args.all:
        # Get stocks that have data
        available = db.list_stocks()
        # Filter to S&P 500
        symbols = [s for s in DEFAULT_US_STOCKS if s in available]
        if len(symbols) < len(DEFAULT_US_STOCKS):
            missing = set(DEFAULT_US_STOCKS) - set(symbols)
            print(f"Warning: {len(missing)} S&P 500 stocks not in database")
            print(f"Run: python scripts/download_sp500_data.py --all")
    elif args.symbols:
        symbols = args.symbols
    else:
        # Default: first 10 that have data
        available = db.list_stocks()
        symbols = [s for s in DEFAULT_US_STOCKS[:10] if s in available]
    
    if args.limit:
        symbols = symbols[:args.limit]
    
    if not symbols:
        print("No symbols to process!")
        print(f"Run first: python scripts/download_sp500_data.py --all")
        return
    
    # Resume: filter already processed
    if args.resume:
        processed_dir = args.library_dir / "processed"
        if processed_dir.exists():
            done = {d.name for d in processed_dir.iterdir() if d.is_dir()}
            symbols = [s for s in symbols if s not in done]
            print(f"Resuming: {len(symbols)} symbols remaining")
    
    print("=" * 70)
    print("OHLC Bar Image Library Builder")
    print("=" * 70)
    print(f"Symbols to process: {len(symbols)}")
    print("=" * 70)
    
    # Build
    result = build_library(
        symbols=symbols,
        data_dir=args.data_dir,
        library_dir=args.library_dir,
        window_size=args.window,
        stride=args.stride,
        include_volume=args.volume,
    )
    
    # Summarize
    summarize(args.library_dir)
    
    print("\n" + "=" * 70)
    print("Build Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
