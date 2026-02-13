"""
US Stock Universe Manager.

Fetches comprehensive US common stock lists from NASDAQ Trader registry.
Filters to common stocks only (no ETFs, warrants, preferred shares, etc.).

Usage:
    from core.stock_universe import USStockUniverse

    universe = USStockUniverse()
    stocks = universe.get_common_stocks()
    print(f"Found {len(stocks)} common stocks")
"""

import re
import logging
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Bundled universe file
DATA_DIR = Path(__file__).parent / "data"
BUNDLED_UNIVERSE_FILE = DATA_DIR / "us_stock_universe.csv"

# NASDAQ Trader HTTP endpoints
NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"


class USStockUniverse:
    """Manage the universe of US common stocks."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_nasdaq_traded(self) -> Optional[pd.DataFrame]:
        """
        Fetch the official NASDAQ traded securities list.

        Returns DataFrame with all listed securities (~8000+).
        Source: nasdaqtrader.com pipe-delimited text file.
        """
        try:
            resp = requests.get(NASDAQ_TRADED_URL, timeout=30)
            resp.raise_for_status()
            text = resp.text

            # Parse pipe-delimited file
            df = pd.read_csv(StringIO(text), sep="|")

            # Last row is typically a timestamp/footer — drop it
            if df.iloc[-1].isna().sum() > len(df.columns) // 2:
                df = df.iloc[:-1]

            # Standardize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]

            logger.info(f"Fetched {len(df)} securities from NASDAQ Trader")
            return df

        except Exception as e:
            logger.warning(f"Failed to fetch NASDAQ traded list: {e}")
            return None

    def _filter_common_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter raw NASDAQ traded securities to common stocks only.

        Removes: ETFs, test issues, warrants, preferred shares,
                 rights, units, and OTC securities.
        """
        original_count = len(df)

        # Column names vary slightly — find them
        etf_col = [c for c in df.columns if "ETF" in c.upper()]
        test_col = [c for c in df.columns if "TEST" in c.upper()]
        exchange_col = [c for c in df.columns if "LISTING" in c.upper() and "EXCHANGE" in c.upper()]
        symbol_col = "Symbol" if "Symbol" in df.columns else "NASDAQ Symbol"
        name_col = [c for c in df.columns if "SECURITY" in c.upper() and "NAME" in c.upper()]

        # Apply filters
        if etf_col:
            df = df[df[etf_col[0]].astype(str).str.strip() == "N"]

        if test_col:
            df = df[df[test_col[0]].astype(str).str.strip() == "N"]

        # Keep major exchanges: N=NYSE, Q=NASDAQ, A=AMEX
        if exchange_col:
            df = df[df[exchange_col[0]].astype(str).str.strip().isin(["N", "Q", "A"])]

        # Filter symbol patterns — remove non-common shares
        symbols = df[symbol_col].astype(str)
        # Remove symbols with special chars (preferred: $, ., #, /)
        mask_clean = ~symbols.str.contains(r'[$#/]', regex=True)
        # Allow dots for tickers like BRK.B, BF.B (single dot followed by letter)
        # but exclude multi-dot or dot-number patterns
        has_dot = symbols.str.contains(r'\.', regex=True)
        dot_ok = symbols.str.match(r'^[A-Z]+\.[A-Z]$', na=False)
        mask_clean = mask_clean & (~has_dot | dot_ok)
        df = df[mask_clean]

        # Filter by security name — remove non-equity securities
        if name_col:
            name_series = df[name_col[0]].astype(str)
            exclude_patterns = [
                r"(?i)\bwarrant",
                r"(?i)\bpreferred\b",
                r"(?i)\brights?\b",
                r"(?i)\bunits?\b",
                r"(?i)\bETF\b",
                r"(?i)\bETN\b",
                r"(?i)\bfund\b",
                r"(?i)\bnotes?\b.*\b20\d{2}\b",  # Debt notes with maturity
                r"(?i)\bdebenture",
                r"(?i)\bconvertible\b.*\bnote",
            ]
            for pattern in exclude_patterns:
                name_series_current = df[name_col[0]].astype(str)
                mask = ~name_series_current.str.contains(pattern, regex=True, na=False)
                df = df[mask]

        logger.info(f"Filtered {original_count} -> {len(df)} common stocks")
        return df

    def get_common_stocks(self) -> pd.DataFrame:
        """
        Get a comprehensive list of US common stocks.

        Returns DataFrame with columns: symbol, name, exchange.
        Tries online NASDAQ source first, falls back to bundled CSV.
        """
        # Try online source
        raw = self.fetch_nasdaq_traded()
        if raw is not None and len(raw) > 100:
            filtered = self._filter_common_stocks(raw)

            # Standardize output columns
            symbol_col = "Symbol" if "Symbol" in filtered.columns else "NASDAQ Symbol"
            name_col = [c for c in filtered.columns if "SECURITY" in c.upper() and "NAME" in c.upper()]
            exchange_col = [c for c in filtered.columns if "LISTING" in c.upper() and "EXCHANGE" in c.upper()]

            result = pd.DataFrame({
                "symbol": filtered[symbol_col].str.strip().values,
                "name": filtered[name_col[0]].str.strip().values if name_col else "",
                "exchange": filtered[exchange_col[0]].str.strip().values if exchange_col else "",
            })

            # Drop rows with NaN symbols
            result = result.dropna(subset=["symbol"])
            result = result[result["symbol"].str.strip().str.len() > 0]

            # Map exchange codes to names
            exchange_map = {"N": "NYSE", "Q": "NASDAQ", "A": "AMEX"}
            result["exchange"] = result["exchange"].map(exchange_map).fillna(result["exchange"])

            # Sort by symbol
            result = result.sort_values("symbol").reset_index(drop=True)

            # Save as bundled fallback
            self.save_universe(result)

            return result

        # Fallback to bundled CSV
        return self.load_bundled_universe()

    def load_bundled_universe(self) -> pd.DataFrame:
        """Load the bundled static CSV as fallback."""
        if BUNDLED_UNIVERSE_FILE.exists():
            df = pd.read_csv(BUNDLED_UNIVERSE_FILE)
            logger.info(f"Loaded {len(df)} stocks from bundled universe")
            return df

        # Ultimate fallback: S&P 500 list
        logger.warning("No universe file found, falling back to S&P 500")
        from .tradingview_fetcher import DEFAULT_US_STOCKS
        return pd.DataFrame({
            "symbol": DEFAULT_US_STOCKS,
            "name": "",
            "exchange": "",
        })

    def save_universe(self, df: pd.DataFrame, path: Optional[Path] = None):
        """Save universe to CSV for version control."""
        path = path or BUNDLED_UNIVERSE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} stocks to {path}")

    def get_ticker_list(self, tier: str = "all") -> List[str]:
        """
        Get ticker list by tier.

        Args:
            tier: "all" (~5000), "sp500" (~503)

        Returns:
            List of ticker symbols.
        """
        if tier == "sp500":
            from .tradingview_fetcher import DEFAULT_US_STOCKS
            return list(DEFAULT_US_STOCKS)

        df = self.get_common_stocks()
        return df["symbol"].tolist()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    universe = USStockUniverse()
    stocks = universe.get_common_stocks()
    print(f"Total common stocks: {len(stocks)}")
    print(f"\nExchange distribution:")
    print(stocks["exchange"].value_counts())
    print(f"\nSample tickers: {stocks['symbol'].head(20).tolist()}")
