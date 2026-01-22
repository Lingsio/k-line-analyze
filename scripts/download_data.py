"""
Download historical stock data for indexing.

This script downloads OHLCV data from various sources and saves it locally.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import json

from app.services.data_fetcher import DataFetcher
from app.config import settings


# Comprehensive stock lists
STOCK_LISTS = {
    'us': [
        # Mega caps
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        # Tech
        'AMD', 'INTC', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'IBM',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY',
        # Consumer
        'WMT', 'HD', 'COST', 'NKE', 'SBUX', 'MCD', 'DIS',
        # Energy
        'XOM', 'CVX', 'COP',
        # Industrial
        'CAT', 'BA', 'GE', 'HON',
    ],
    'tw': [
        # Semiconductors
        '2330', '2454', '2303', '3711', '2344', '2379',
        # Electronics
        '2317', '2382', '2357', '2308', '2412',
        # Finance
        '2881', '2882', '2891', '2886', '2884',
        # Traditional
        '1301', '1303', '2002', '1326',
    ],
    'cn': [
        # Blue chips
        '600519', '000858', '601318', '600036', '000001',
        # Tech
        '000333', '002415', '300750',
        # Consumer
        '000568', '603259',
        # Finance
        '601398', '601288',
    ],
    'hk': [
        # Tech
        '0700', '9988', '9618', '3690',
        # Finance
        '0005', '0939', '1398', '2318',
        # Property
        '0016', '0001', '0012',
        # Others
        '0941', '0883', '0388',
    ],
    'crypto': [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT',
        'MATIC', 'AVAX', 'LINK', 'UNI',
    ],
}


async def download_symbol(
    fetcher: DataFetcher,
    symbol: str,
    market: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> bool:
    """Download data for a single symbol."""
    try:
        df = await fetcher.fetch_ohlcv(
            symbol=symbol,
            market=market,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or df.empty:
            return False

        # Save to parquet for efficiency
        output_path = output_dir / market / f"{symbol}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path)
        return True

    except Exception as e:
        print(f"Error downloading {market}:{symbol}: {e}")
        return False


async def download_market(
    market: str,
    symbols: list,
    start_date: str,
    end_date: str,
    output_dir: Path,
):
    """Download all symbols for a market."""
    fetcher = DataFetcher()
    success_count = 0

    for symbol in tqdm(symbols, desc=f"Downloading {market.upper()}"):
        success = await download_symbol(
            fetcher=fetcher,
            symbol=symbol,
            market=market,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )
        if success:
            success_count += 1

        # Rate limiting
        await asyncio.sleep(0.5)

    return success_count


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download historical stock data')
    parser.add_argument('--markets', nargs='+', default=None,
                        help='Markets to download (us, tw, cn, hk, crypto)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Set defaults
    markets = args.markets or list(STOCK_LISTS.keys())

    if args.end_date:
        end_date = args.end_date
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if args.start_date:
        start_date = args.start_date
    else:
        start_date = (datetime.now() - timedelta(days=365*20)).strftime('%Y-%m-%d')

    output_dir = Path(args.output) if args.output else settings.DATA_DIR / 'raw'

    print("=" * 60)
    print("K-Line Data Downloader")
    print("=" * 60)
    print(f"Markets: {markets}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Download data for each market
    total_success = 0
    total_attempted = 0

    for market in markets:
        symbols = STOCK_LISTS.get(market, [])
        if not symbols:
            continue

        print(f"\nDownloading {market.upper()} market ({len(symbols)} symbols)...")
        total_attempted += len(symbols)

        success = await download_market(
            market=market,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )

        total_success += success
        print(f"Successfully downloaded: {success}/{len(symbols)}")

    # Save download info
    info = {
        'download_date': datetime.now().isoformat(),
        'date_range': {'start': start_date, 'end': end_date},
        'markets': markets,
        'total_symbols': total_attempted,
        'successful_downloads': total_success,
    }

    info_path = output_dir / 'download_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Total: {total_success}/{total_attempted} symbols downloaded")
    print(f"Data saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
