"""
Multi-source data fetcher for stock and cryptocurrency data.

Supported markets:
- US: Yahoo Finance
- TW: Yahoo Finance (with .TW/.TWO suffix)
- CN: AKShare
- HK: Yahoo Finance (with .HK suffix)
- Crypto: CCXT (Binance)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import asyncio
from functools import lru_cache


class DataFetcher:
    """Unified data fetcher for multiple markets."""

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # (data, timestamp)
        self._cache_ttl = 3600  # 1 hour

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run blocking function in thread pool."""
        import functools
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: str = "us",
        start_date: str = None,
        end_date: str = None,
        period: str = "daily",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Stock/crypto symbol
            market: Market identifier (us, tw, cn, hk, crypto)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: 'daily' or 'weekly'

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        cache_key = f"{market}:{symbol}:{start_date}:{end_date}:{period}"

        # Check cache
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data.copy()

        # Fetch based on market
        # Note: The sub-methods (_fetch_yahoo, etc) are currently marked async but contain blocking calls.
        # We should update them to be sync and run them via _run_in_executor, 
        # OR update them to wrap their internal blocking calls.
        # To minimize changes, I will wrap the calls here if possible, but they are defined as async.
        # Let's check _fetch_china definition. It was async def _fetch_china(...)
        # But inside it calls blocking code.
        # "async def" does NOT make code async if it doesn't await. It just returns a coroutine.
        # If I await it, it runs synchronously.
        
        # Correct fix: Change _fetch_china etc to be synchronous, then await _run_in_executor(self._fetch_china, ...)
        # BUT changing signature breaks interface if used elsewhere? 
        # Only used internally here?
        
        # Let's keep them async def, but inside them, wrap the blocking part? 
        # No, easier: call them directly. 
        # Wait, if they are `async def`, I must `await` them.
        # I should change the implementation of `_fetch_china` to wrap the blocking call.
        
        if market == "us":
            df = await self._fetch_yahoo(symbol, start_date, end_date, period)
        elif market == "tw":
            df = await self._fetch_taiwan(symbol, start_date, end_date, period)
        elif market == "cn":
            df = await self._fetch_china(symbol, start_date, end_date, period)
        elif market == "hk":
            df = await self._fetch_hongkong(symbol, start_date, end_date, period)
        elif market == "crypto":
            # Crypto uses CCXT which supports async but here might be sync or wrapped?
            # Existing code used 'await self._fetch_crypto'.
            # Let's check _fetch_crypto. 
            # If it uses standard CCXT sync, it blocks.
            df = await self._fetch_crypto(symbol, start_date, end_date, period)
        else:
            raise ValueError(f"Unsupported market: {market}")

        if df is not None and not df.empty:
            # Standardize column names
            df = self._standardize_columns(df)
            # Cache the result
            self._cache[cache_key] = (df, datetime.now().timestamp())

        return df

    async def _fetch_yahoo(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            interval = "1d" if period == "daily" else "1wk"

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                return None

            return df

        except Exception as e:
            print(f"Error fetching Yahoo data for {symbol}: {e}")
            return None

    async def _fetch_taiwan(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch Taiwan stock data."""
        try:
            import yfinance as yf

            # Taiwan stocks use .TW (TWSE) or .TWO (TPEx) suffix
            if not symbol.endswith((".TW", ".TWO")):
                # Try TWSE first
                tw_symbol = f"{symbol}.TW"
            else:
                tw_symbol = symbol

            ticker = yf.Ticker(tw_symbol)
            interval = "1d" if period == "daily" else "1wk"

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                # Try TPEx
                tw_symbol = f"{symbol}.TWO"
                ticker = yf.Ticker(tw_symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)

            return df if not df.empty else None

        except Exception as e:
            print(f"Error fetching Taiwan data for {symbol}: {e}")
            return None

    async def _fetch_china(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch China A-share data via AKShare."""
        try:
            import akshare as ak

            import akshare as ak
            
            # Handle explicit prefixes (sh/sz)
            clean_symbol = symbol.lower()
            if clean_symbol.startswith(("sh", "sz")):
                prefix = clean_symbol[:2]
                code = clean_symbol[2:]
            else:
                # Default inference
                code = symbol
                prefix = "sh" if symbol.startswith("6") else "sz"
            
            full_symbol = f"{prefix}{code}"

            # Wrapper for blocking calls
            def fetch_stock_sync():
                return ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily" if period == "daily" else "weekly",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
            
            def fetch_index_sync():
                 return ak.stock_zh_index_daily(symbol=full_symbol)

            # 1. Try fetching as standard stock
            try:
                # Let's try fetching stock first ONLY if it's not a likely index request
                is_likely_index = (prefix == "sh" and code == "000001") or (prefix == "sz" and code == "399001")
                
                df = None
                if not is_likely_index:
                    # Run potentially blocking AKShare call in executor
                    df = await self._run_in_executor(fetch_stock_sync)
                
                # If not found or empty, OR if it's a likely index request, try Index API
                if (df is None or df.empty) or is_likely_index:
                    # Try Index API
                    index_df = await self._run_in_executor(fetch_index_sync)
                    
                    if index_df is not None and not index_df.empty:
                        # Index data columns usually: date, open, high, low, close, volume
                        df = index_df
                        # Make sure date filtering is applied as index api might return all history
                        df["date"] = pd.to_datetime(df["date"])
                        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
                        df = df.loc[mask]

                if df is None or df.empty:
                    return None

                # Rename columns
                # AKShare index columns might be consistent with stock, but let's be safe
                rename_map = {
                    "日期": "date", "Date": "date",
                    "开盘": "open", "Open": "open",
                    "收盘": "close", "Close": "close",
                    "最高": "high", "High": "high",
                    "最低": "low", "Low": "low",
                    "成交量": "volume", "Volume": "volume",
                    "成交额": "amount", "Amount": "amount"
                }
                
                # Check which columns exist
                current_cols = df.columns.tolist()
                new_cols = {}
                for c in current_cols:
                    if c in rename_map:
                        new_cols[c] = rename_map[c]
                
                df = df.rename(columns=new_cols)
                
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                
                return df

            except KeyError:
                # If symbol not found in stock API, pass to try logic below or return None
                 pass
            except Exception as inner_e:
                print(f"AKShare internal error: {inner_e}")
                pass
            
            return None

        except Exception as e:
            print(f"Error fetching China data for {symbol}: {e}")
            # Fallback to Yahoo Finance with .SS or .SZ suffix
            return await self._fetch_china_yahoo(symbol, start_date, end_date, period)

    async def _fetch_china_yahoo(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fallback: Fetch China data from Yahoo Finance."""
        try:
            import yfinance as yf

            # Shanghai: .SS, Shenzhen: .SZ
            if symbol.startswith("6"):
                yahoo_symbol = f"{symbol}.SS"
            else:
                yahoo_symbol = f"{symbol}.SZ"

            ticker = yf.Ticker(yahoo_symbol)
            interval = "1d" if period == "daily" else "1wk"

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            return df if not df.empty else None

        except Exception as e:
            print(f"Error fetching China Yahoo data for {symbol}: {e}")
            return None

    async def _fetch_hongkong(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch Hong Kong stock data."""
        try:
            import yfinance as yf

            # HK stocks use .HK suffix, pad with zeros to 4 digits
            if not symbol.endswith(".HK"):
                hk_symbol = f"{symbol.zfill(4)}.HK"
            else:
                hk_symbol = symbol

            ticker = yf.Ticker(hk_symbol)
            interval = "1d" if period == "daily" else "1wk"

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            return df if not df.empty else None

        except Exception as e:
            print(f"Error fetching HK data for {symbol}: {e}")
            return None

    async def _fetch_crypto(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data via CCXT."""
        try:
            import ccxt

            # Default to Binance
            exchange = ccxt.binance({"enableRateLimit": True})

            # Format symbol (e.g., BTC -> BTC/USDT)
            if "/" not in symbol:
                symbol = f"{symbol}/USDT"

            timeframe = "1d" if period == "daily" else "1w"

            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

            # Fetch OHLCV
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=start_ts, limit=1000
            )

            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("date")
            df = df.drop("timestamp", axis=1)

            # Filter by end date
            df = df[df.index <= end_date]

            return df

        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            # Fallback to Yahoo Finance
            return await self._fetch_crypto_yahoo(symbol, start_date, end_date, period)

    async def _fetch_crypto_yahoo(
        self, symbol: str, start_date: str, end_date: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fallback: Fetch crypto data from Yahoo Finance."""
        try:
            import yfinance as yf

            # Yahoo uses format like BTC-USD
            if "/" in symbol:
                base, quote = symbol.split("/")
                yahoo_symbol = f"{base}-{quote}"
            else:
                yahoo_symbol = f"{symbol}-USD"

            ticker = yf.Ticker(yahoo_symbol)
            interval = "1d" if period == "daily" else "1wk"

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            return df if not df.empty else None

        except Exception as e:
            print(f"Error fetching crypto Yahoo data for {symbol}: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame column names to lowercase."""
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                # Try to find similar column
                for c in df.columns:
                    if col in c.lower():
                        df[col] = df[c]
                        break

        # Keep only required columns
        df = df[[c for c in required_cols if c in df.columns]]

        return df

    async def get_stock_info(self, symbol: str, market: str) -> Dict[str, Any]:
        """Get basic stock information."""
        try:
            import yfinance as yf

            # Adjust symbol based on market
            if market == "tw" and not symbol.endswith((".TW", ".TWO")):
                symbol = f"{symbol}.TW"
            elif market == "hk" and not symbol.endswith(".HK"):
                symbol = f"{symbol.zfill(4)}.HK"
            elif market == "cn":
                if symbol.startswith("6"):
                    symbol = f"{symbol}.SS"
                else:
                    symbol = f"{symbol}.SZ"
            elif market == "crypto" and "-" not in symbol:
                symbol = f"{symbol}-USD"

            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "name": info.get("longName", info.get("shortName", symbol)),
                "market": market,
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            }

        except Exception as e:
            return {
                "symbol": symbol,
                "name": symbol,
                "market": market,
                "error": str(e),
            }

    async def fetch_batch(
        self,
        symbols: list,
        market: str,
        start_date: str,
        end_date: str,
        period: str = "daily",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently."""
        tasks = [
            self.fetch_ohlcv(symbol, market, start_date, end_date, period)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result
            for symbol, result in zip(symbols, results)
            if isinstance(result, pd.DataFrame)
        }
