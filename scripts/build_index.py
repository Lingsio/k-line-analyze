"""
Build FAISS index from historical K-line data.

This script:
1. Downloads historical data for configured markets
2. Generates K-line images and extracts embeddings
3. Builds and saves the FAISS index
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import json

from app.services.data_fetcher import DataFetcher
from app.services.preprocessor import KLinePreprocessor
from app.services.feature_extractor import FeatureExtractor
from app.services.similarity_search import SimilaritySearchEngine
from app.config import settings


# Stock lists (subset for demo)
STOCK_LISTS = {
    'us': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'NFLX', 'INTC'],
    'tw': ['2330', '2317', '2454', '2412', '2308'],
    'cn': ['600519', '000858', '601318'],
    'hk': ['0700', '9988', '0005'],
    'crypto': ['BTC', 'ETH', 'BNB'],
}


async def download_market_data(
    market: str,
    symbols: list,
    start_date: str,
    end_date: str,
) -> dict:
    """Download data for all symbols in a market."""
    fetcher = DataFetcher()
    data = {}

    for symbol in tqdm(symbols, desc=f"Downloading {market}"):
        try:
            df = await fetcher.fetch_ohlcv(
                symbol=symbol,
                market=market,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and len(df) > 60:
                data[symbol] = df
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")

    return data


def extract_windows(
    df,
    symbol: str,
    market: str,
    preprocessor: KLinePreprocessor,
    feature_extractor: FeatureExtractor,
    window_size: int = 60,
    step: int = 5,
) -> list:
    """Extract embeddings for all windows in a DataFrame."""
    windows = preprocessor.create_windows(df, window_size, step)
    results = []

    for window_df, start_idx, end_idx in windows:
        try:
            # Normalize and extract embedding
            normalized = preprocessor.normalize(window_df)
            embedding = feature_extractor.extract_from_dataframe(normalized)

            # Get dates
            start_date = window_df.index[0]
            end_date = window_df.index[-1]

            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')

            results.append({
                'symbol': symbol,
                'market': market,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'embedding': embedding,
                'series': normalized['close'].values.tolist(),
            })
        except Exception as e:
            continue

    return results


async def build_index(
    markets: list = None,
    start_date: str = None,
    end_date: str = None,
    window_size: int = 60,
    step: int = 5,
):
    """Build the complete FAISS index."""
    print("=" * 60)
    print("K-Line Pattern Index Builder")
    print("=" * 60)

    markets = markets or list(STOCK_LISTS.keys())

    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    print(f"Markets: {markets}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Window size: {window_size} days")
    print(f"Step: {step} days")
    print("=" * 60)

    # Initialize components
    preprocessor = KLinePreprocessor()
    feature_extractor = FeatureExtractor()
    search_engine = SimilaritySearchEngine(dimension=settings.EMBEDDING_DIM)

    all_embeddings = []
    all_metadata = []

    # Process each market
    for market in markets:
        symbols = STOCK_LISTS.get(market, [])
        if not symbols:
            continue

        print(f"\nProcessing {market.upper()} market ({len(symbols)} symbols)...")

        # Download data
        market_data = await download_market_data(
            market=market,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        print(f"Downloaded data for {len(market_data)} symbols")

        # Extract embeddings
        for symbol, df in tqdm(market_data.items(), desc="Extracting features"):
            windows = extract_windows(
                df=df,
                symbol=symbol,
                market=market,
                preprocessor=preprocessor,
                feature_extractor=feature_extractor,
                window_size=window_size,
                step=step,
            )

            for w in windows:
                all_embeddings.append(w['embedding'])
                all_metadata.append({
                    'symbol': w['symbol'],
                    'market': w['market'],
                    'start_date': w['start_date'],
                    'end_date': w['end_date'],
                    'series': w['series'],
                })

    print(f"\nTotal patterns extracted: {len(all_embeddings)}")

    if not all_embeddings:
        print("No embeddings extracted. Exiting.")
        return

    # Build index
    print("\nBuilding FAISS index...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    search_engine.add_vectors(embeddings_array, all_metadata)

    # Save index
    index_dir = settings.FAISS_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    search_engine.save_index(str(index_dir))

    print(f"\nIndex saved to: {index_dir}")
    print(f"Total vectors: {search_engine.get_total_vectors()}")
    print(f"Markets indexed: {search_engine.get_indexed_markets()}")

    # Save build info
    build_info = {
        'build_date': datetime.now().isoformat(),
        'markets': markets,
        'date_range': {'start': start_date, 'end': end_date},
        'window_size': window_size,
        'step': step,
        'total_vectors': search_engine.get_total_vectors(),
    }

    with open(index_dir / 'build_info.json', 'w') as f:
        json.dump(build_info, f, indent=2)

    print("\n" + "=" * 60)
    print("Index build complete!")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build K-line pattern index')
    parser.add_argument('--markets', nargs='+', default=None,
                        help='Markets to index (us, tw, cn, hk, crypto)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--window-size', type=int, default=60,
                        help='Window size in days')
    parser.add_argument('--step', type=int, default=5,
                        help='Step size between windows')

    args = parser.parse_args()

    asyncio.run(build_index(
        markets=args.markets,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=args.window_size,
        step=args.step,
    ))


if __name__ == '__main__':
    main()
