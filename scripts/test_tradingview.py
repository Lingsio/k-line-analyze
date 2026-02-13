"""
Test TradingView data fetching functionality
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tradingview_fetcher import TradingViewFetcher, USStockDatabase, DEFAULT_US_STOCKS


def test_fetcher():
    """Test TradingViewFetcher"""
    print("=" * 60)
    print("Testing TradingViewFetcher")
    print("=" * 60)
    
    fetcher = TradingViewFetcher()
    
    # Test AAPL
    print("\n1. Testing AAPL daily data...")
    df = fetcher.fetch_ohlcv("AAPL", n_bars=100)
    if df is not None:
        print(f"   [OK] Success: {len(df)} rows")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(df.head(3))
    else:
        print("   [FAIL] Failed")
    
    # Test MSFT historical data
    print("\n2. Testing MSFT historical data (2020-2023)...")
    df = fetcher.fetch_history("MSFT", "2020-01-01", "2023-12-31")
    if df is not None:
        print(f"   [OK] Success: {len(df)} rows")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    else:
        print("   [FAIL] Failed")
    
    # Test different time intervals
    print("\n3. Testing different intervals...")
    for interval in ["1d", "1W"]:
        df = fetcher.fetch_ohlcv("GOOGL", interval=interval, n_bars=100)
        if df is not None:
            print(f"   [OK] {interval}: {len(df)} bars")
        else:
            print(f"   [FAIL] {interval}: Failed")
    
    print("\n" + "=" * 60)


def test_database():
    """Test USStockDatabase"""
    print("=" * 60)
    print("Testing USStockDatabase")
    print("=" * 60)
    
    # Use temp directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        db = USStockDatabase(data_dir=tmpdir)
        
        print("\n1. Testing update_stock (AAPL)...")
        success = db.update_stock("AAPL", start_date="2023-01-01")
        print(f"   [{'OK' if success else 'FAIL'}]")
        
        print("\n2. Testing load_stock...")
        df = db.load_stock("AAPL")
        if df is not None:
            print(f"   [OK] Loaded: {len(df)} rows")
            print(df.tail(3))
        else:
            print("   [FAIL] Failed to load")
        
        print("\n3. Testing list_stocks...")
        stocks = db.list_stocks()
        print(f"   [OK] Stocks in database: {stocks}")
        
        print("\n4. Testing metadata...")
        info = db.get_stock_info("AAPL")
        if info:
            print(f"   [OK] Metadata: {info}")
        else:
            print("   [FAIL] No metadata")
    
    print("\n" + "=" * 60)


def test_data_fetcher_integration():
    """Test DataFetcher integration"""
    print("=" * 60)
    print("Testing DataFetcher with TradingView")
    print("=" * 60)
    
    from core.data_fetcher import DataFetcher
    import asyncio
    
    async def test():
        fetcher = DataFetcher(use_tradingview=True)
        
        print("\n1. Testing US stock fetch...")
        df = await fetcher.fetch_ohlcv("AAPL", market="us", start_date="2024-01-01")
        if df is not None:
            print(f"   [OK] Success: {len(df)} rows")
            print(df.tail(3))
        else:
            print("   [FAIL] Failed")
        
        print("\n2. Testing batch fetch...")
        results = await fetcher.fetch_batch(
            ["MSFT", "GOOGL", "AMZN"],
            market="us",
            start_date="2024-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )
        print(f"   [OK] Fetched {len(results)} stocks: {list(results.keys())}")
    
    asyncio.run(test())
    print("\n" + "=" * 60)


def main():
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 15 + "TradingView Test Suite" + " " * 21 + "|")
    print("+" + "=" * 58 + "+")
    print()
    
    try:
        test_fetcher()
    except Exception as e:
        print(f"Fetcher test failed: {e}")
    
    try:
        test_database()
    except Exception as e:
        print(f"Database test failed: {e}")
    
    try:
        test_data_fetcher_integration()
    except Exception as e:
        print(f"DataFetcher integration test failed: {e}")
    
    print("\n[OK] All tests completed!")


if __name__ == "__main__":
    main()
