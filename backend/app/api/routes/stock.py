from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import date, datetime, timedelta
from pydantic import BaseModel

from app.services.data_fetcher import DataFetcher

router = APIRouter()
data_fetcher = DataFetcher()


class OHLCVData(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class KLineResponse(BaseModel):
    symbol: str
    market: str
    period: str
    data: List[OHLCVData]
    count: int


class MarketInfo(BaseModel):
    market: str
    name: str
    description: str
    supported: bool


@router.get("/stock/{symbol}/kline", response_model=KLineResponse)
async def get_kline_data(
    symbol: str,
    market: str = Query("us", description="Market: us, tw, cn, hk, crypto"),
    period: str = Query("daily", description="Period: daily, weekly"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Get K-line (OHLCV) data for a stock/crypto.

    Markets:
    - us: US stocks (via Yahoo Finance)
    - tw: Taiwan stocks (via FinMind or Yahoo)
    - cn: China A-shares (via AKShare)
    - hk: Hong Kong stocks (via AKShare or Yahoo)
    - crypto: Cryptocurrency (via CCXT)
    """
    try:
        # Default date range: last 1 year
        if not end_date:
            end_date = date.today().isoformat()
        if not start_date:
            start_date = (date.today() - timedelta(days=365)).isoformat()

        df = await data_fetcher.fetch_ohlcv(
            symbol=symbol,
            market=market,
            start_date=start_date,
            end_date=end_date,
            period=period,
        )

        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {symbol} in {market} market",
            )

        # Convert DataFrame to list of dicts
        data = []
        for idx, row in df.iterrows():
            data.append(
                OHLCVData(
                    date=idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    open=round(float(row["open"]), 4),
                    high=round(float(row["high"]), 4),
                    low=round(float(row["low"]), 4),
                    close=round(float(row["close"]), 4),
                    volume=float(row["volume"]),
                )
            )

        return KLineResponse(
            symbol=symbol.upper(),
            market=market,
            period=period,
            data=data,
            count=len(data),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets", response_model=List[MarketInfo])
async def get_supported_markets():
    """Get list of supported markets."""
    return [
        MarketInfo(
            market="us",
            name="US Stocks",
            description="NYSE, NASDAQ via Yahoo Finance",
            supported=True,
        ),
        MarketInfo(
            market="tw",
            name="Taiwan Stocks",
            description="TWSE, TPEx via Yahoo Finance",
            supported=True,
        ),
        MarketInfo(
            market="cn",
            name="China A-Shares",
            description="SSE, SZSE via AKShare",
            supported=True,
        ),
        MarketInfo(
            market="hk",
            name="Hong Kong Stocks",
            description="HKEX via Yahoo Finance",
            supported=True,
        ),
        MarketInfo(
            market="crypto",
            name="Cryptocurrency",
            description="Major exchanges via CCXT",
            supported=True,
        ),
    ]


@router.get("/stock/{symbol}/info")
async def get_stock_info(
    symbol: str,
    market: str = Query("us", description="Market: us, tw, cn, hk, crypto"),
):
    """Get basic stock information."""
    try:
        info = await data_fetcher.get_stock_info(symbol, market)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
