"""
Build sector/industry groupings for all US stocks using yfinance.

Fetches GICS sector info from yfinance for each stock,
then creates a sector grouping JSON file for training.

Usage:
    python scripts/build_sector_groups.py
    python scripts/build_sector_groups.py --resume
    python scripts/build_sector_groups.py --info
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tradingview_fetcher import USStockDatabase

# Try yfinance
import yfinance as yf


def fetch_sector_info(symbol: str) -> dict:
    """Fetch sector/industry info for a single stock from yfinance."""
    try:
        yf_sym = symbol.replace(".", "-")
        ticker = yf.Ticker(yf_sym)
        info = ticker.info
        return {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "name": info.get("shortName", ""),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {
            "sector": "Unknown",
            "industry": "Unknown",
            "name": "",
            "market_cap": 0,
        }


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_groups_from_sectors(sector_data: dict, min_stocks: int = 20) -> dict:
    """
    Build training groups from sector data.

    Uses GICS sectors as primary grouping.
    Sectors with too few stocks are merged into 'Other'.
    """
    # Group by sector
    sector_stocks = {}
    for symbol, info in sector_data.items():
        sector = info.get("sector", "Unknown")
        if sector in (None, "", "Unknown", "N/A"):
            sector = "Other"
        if sector not in sector_stocks:
            sector_stocks[sector] = []
        sector_stocks[sector].append(symbol)

    # Merge small sectors into 'Other'
    groups = {}
    other_stocks = []
    for sector, stocks in sorted(sector_stocks.items()):
        if len(stocks) >= min_stocks and sector != "Other":
            # Clean sector name for use as dict key
            key = sector.replace(" ", "_")
            groups[key] = {
                "name": sector,
                "stocks": sorted(stocks),
                "description": f"{sector} ({len(stocks)} stocks)",
            }
        else:
            other_stocks.extend(stocks)

    if other_stocks:
        groups["Other"] = {
            "name": "Other",
            "stocks": sorted(other_stocks),
            "description": f"Uncategorized and small sectors ({len(other_stocks)} stocks)",
        }

    return groups


def main():
    parser = argparse.ArgumentParser(description="Build sector groupings for US stocks")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from cached sector data")
    parser.add_argument("--info", action="store_true", help="Show existing groupings")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between yfinance requests")
    parser.add_argument("--limit", type=int, help="Limit stocks to process")
    parser.add_argument("--min-stocks", type=int, default=20, help="Min stocks per sector group")

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = Path(__file__).parent.parent / "data" / "raw" / "us"

    output_file = Path(__file__).parent.parent / "data" / "us_stock_groups_full.json"
    cache_file = Path(__file__).parent.parent / "data" / "sector_cache.json"

    # Info mode
    if args.info:
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("=" * 70)
            print("US Stock Sector Groups")
            print("=" * 70)
            total = 0
            for key, group in data["groups"].items():
                n = len(group["stocks"])
                total += n
                print(f"  {key:<30} {n:>5} stocks  {group['name']}")
            print("-" * 70)
            print(f"  {'TOTAL':<30} {total:>5} stocks")
            print(f"  Groups: {len(data['groups'])}")
            print("=" * 70)
        else:
            print("No groupings found. Run: python scripts/build_sector_groups.py")
        return

    # Load database
    db = USStockDatabase(data_dir=args.data_dir)
    all_stocks = sorted(db.list_stocks())

    if args.limit:
        all_stocks = all_stocks[:args.limit]

    print(f"Total stocks in database: {len(all_stocks)}")

    # Load cache
    sector_cache = load_cache(cache_file) if args.resume else {}
    print(f"Cached sector info: {len(sector_cache)} stocks")

    # Fetch missing sector info
    to_fetch = [s for s in all_stocks if s not in sector_cache]
    print(f"Need to fetch: {len(to_fetch)} stocks")

    if to_fetch:
        print("\nFetching sector info from yfinance...")
        pbar = tqdm(to_fetch, desc="Fetching", unit="stock")

        for symbol in pbar:
            info = fetch_sector_info(symbol)
            sector_cache[symbol] = info
            pbar.set_postfix({"sector": info["sector"][:15] if info["sector"] else "?"})

            time.sleep(args.delay)

            # Checkpoint every 100 stocks
            if len(sector_cache) % 100 == 0:
                save_cache(cache_file, sector_cache)

        save_cache(cache_file, sector_cache)
        print(f"\nSector cache saved: {len(sector_cache)} stocks")

    # Filter to stocks we actually have data for
    valid_cache = {s: sector_cache[s] for s in all_stocks if s in sector_cache}

    # Build groups
    groups = build_groups_from_sectors(valid_cache, min_stocks=args.min_stocks)

    # Statistics
    total_stocks = sum(len(g["stocks"]) for g in groups.values())
    stats = {
        "total_stocks": total_stocks,
        "num_groups": len(groups),
        "avg_stocks_per_group": round(total_stocks / len(groups), 1),
        "built_at": datetime.now().isoformat(),
    }

    # Save
    output_data = {
        "groups": groups,
        "statistics": stats,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 70)
    print("Sector Groups Built")
    print("=" * 70)
    for key, group in groups.items():
        n = len(group["stocks"])
        print(f"  {key:<30} {n:>5} stocks")
    print("-" * 70)
    print(f"  {'TOTAL':<30} {total_stocks:>5} stocks")
    print(f"  Groups: {len(groups)}")
    print(f"\nSaved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
