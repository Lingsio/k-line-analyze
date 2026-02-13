"""
yfinance-based data fetcher for US stocks.

Provides a reliable REST-based alternative to TradingView WebSocket.
Features adaptive rate limiting, retry logic, and column standardization.

Usage:
    from core.yfinance_fetcher import YFinanceFetcher

    fetcher = YFinanceFetcher()
    df = fetcher.fetch_ohlcv("AAPL", start_date="2000-01-01")
"""

import time
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _to_yf_symbol(symbol: str) -> str:
    """Convert standard ticker to yfinance format (BRK.B -> BRK-B)."""
    return symbol.replace(".", "-")


class YFinanceFetcher:
    """
    yfinance data fetcher with adaptive rate limiting.
    """

    def __init__(
        self,
        min_delay: float = 0.3,
        max_delay: float = 5.0,
        adaptive: bool = True,
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.adaptive = adaptive
        self._delay = min_delay
        self._consecutive_errors = 0

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        retry: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a single symbol.

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume],
            timezone-naive DatetimeIndex named 'Date'.
            None if download fails.
        """
        yf_sym = _to_yf_symbol(symbol)

        for attempt in range(retry):
            try:
                self._wait()

                ticker = yf.Ticker(yf_sym)
                kwargs = {"start": start_date, "auto_adjust": True}
                if end_date:
                    kwargs["end"] = end_date

                df = ticker.history(**kwargs)

                if df is None or df.empty:
                    self._on_error()
                    if attempt < retry - 1:
                        continue
                    return None

                # Keep only OHLCV columns
                cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[cols]

                # Remove timezone from index
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index.name = "Date"

                self._on_success()
                return df

            except Exception as e:
                self._on_error()
                if attempt < retry - 1:
                    time.sleep(self._delay * (attempt + 1))
                else:
                    logger.debug(f"Failed to fetch {symbol}: {e}")
                    return None

        return None

    def fetch_history(
        self,
        symbol: str,
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Compatible interface with TradingViewFetcher.fetch_history()."""
        return self.fetch_ohlcv(symbol, start_date=str(start_date), end_date=end_date)

    def _wait(self):
        """Apply rate limiting delay."""
        time.sleep(self._delay)

    def _on_success(self):
        """Reduce delay after successful request."""
        self._consecutive_errors = 0
        if self.adaptive:
            self._delay = max(self.min_delay, self._delay * 0.9)

    def _on_error(self):
        """Increase delay after error."""
        self._consecutive_errors += 1
        if self.adaptive:
            self._delay = min(self.max_delay, self._delay * 1.5)
        # Long pause after many consecutive errors
        if self._consecutive_errors >= 5:
            time.sleep(30)
            self._consecutive_errors = 0
