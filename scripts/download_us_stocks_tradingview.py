"""
Download US stock data using TradingView (tvdatafeed).

Features:
- 支持获取历史数据（1990年至今）
- 自动处理 symbol 映射（AAPL -> NASDAQ:AAPL）
- 支持多种 timeframe（1m, 5m, 1h, 1d, 1W, 1M）
- 自动重试和错误处理
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import time
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import tvdatafeed
try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    print("Installing tvdatafeed...")
    os.system("pip install tvdatafeed -q")
    from tvDatafeed import TvDatafeed, Interval

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "us"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NASDAQ 100 + S&P 500 主要成分股 (共500+只美股)
US_STOCKS = [
    # === 科技板块 (Tech) ===
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC",
    "NFLX", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "AVGO", "QCOM", "TXN", "NOW",
    "SNOW", "ZM", "UBER", "LYFT", "ABNB", "DDOG", "MDB", "CRWD", "OKTA", "NET",
    "DOCU", "SQ", "SHOP", "SPOT", "RBLX", "TWLO", "FSLY", "PLTR", "SNOW", "ZM",
    "PYPL", "V", "MA", "AXP", "COF", "DFS", "SYF", "ALLY", "BAC", "JPM",
    
    # === 半导体 (Semiconductors) ===
    "NVDA", "AMD", "INTC", "QCOM", "TXN", "AVGO", "MU", "LRCX", "AMAT", "KLAC",
    "SNPS", "CDNS", "MRVL", "NXPI", "SWKS", "QRVO", "ON", "MCHP", "MPWR", "RMBS",
    
    # === 金融板块 (Financials) ===
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "BK",
    "STT", "BLK", "BX", "KKR", "APO", "CG", "MSCI", "SPGI", "MCO", "FIS",
    "FISV", "GPN", "FLT", "MA", "V", "AXP", "DFS", "COF", "SYF", "ALLY",
    
    # === 医疗健康 (Healthcare) ===
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
    "GILD", "REGN", "VRTX", "BIIB", "ISRG", "ZTS", "CVS", "CI", "HUM", "ANTM",
    "MDT", "SYK", "BSX", "EW", "DXCM", "ABMD", "ILMN", "IQV", "DHR", "A",
    
    # === 消费品 (Consumer) ===
    "WMT", "HD", "COST", "NKE", "SBUX", "MCD", "DIS", "PG", "KO", "PEP",
    "WMT", "TGT", "COST", "DG", "DLTR", "BBY", "TJX", "ROST", "BURL", "GPS",
    "NKE", "LULU", "UA", "UAA", "DECK", "SKX", "CROX", "TPR", "COH", "EL",
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "GIS", "K", "CPB",
    "HSY", "MDLZ", "KHC", "SJM", "CAG", "TSN", "ADM", "BG", "INGR", "FLO",
    
    # === 能源板块 (Energy) ===
    "XOM", "CVX", "COP", "EOG", "PXD", "MPC", "VLO", "PSX", "WMB", "OKE",
    "KMI", "EPD", "ET", "MPLX", "ENB", "TRP", "OXY", "DVN", "FANG", "MRO",
    "HES", "APA", "OVV", "EQT", "RRC", "AR", "SWN", "CTRA", "CHK", "MUR",
    
    # === 工业板块 (Industrials) ===
    "BA", "CAT", "GE", "RTX", "LMT", "NOC", "GD", "HII", "TDG", "TRU",
    "HON", "MMM", "ITW", "ILL", "SWK", "SNA", "GRMN", "TRMB", "FLS", "PNR",
    "UPS", "FDX", "CSX", "UNP", "NSC", "KSU", "LSTR", "CHRW", "EXPD", "XT",
    
    # === 通信服务 (Communication Services) ===
    "GOOGL", "META", "DIS", "NFLX", "CMCSA", "VZ", "T", "TMUS", "CHTR", "ATVI",
    "EA", "TTWO", "RBLX", "MTCH", "BMBL", "PINS", "SNAP", "TWTR", "SPOT", "IAC",
    
    # === 房地产 (Real Estate) ===
    "AMT", "PLD", "CCI", "EQIX", "DLR", "PSA", "O", "WPC", "NNN", "STOR",
    "SPG", "MAC", "GGP", "KIM", "FRT", "REG", "BXP", "SLG", "VTR", "HCP",
    
    # === 基础材料 (Materials) ===
    "LIN", "APD", "ECL", "SHW", "PPG", "DD", "LYB", "ALB", "SQM", "FMC",
    "NEM", "GOLD", "AEM", "KL", "AU", "ABX", "FCX", "SCCO", "TECK", "VALE",
    
    # === 公用事业 (Utilities) ===
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "FE",
    "PEG", "ETR", "AEE", "CNP", "NI", "WEC", "CMS", "ATO", "LNT", "EVRG",
]

# 去除重复
US_STOCKS = sorted(list(set(US_STOCKS)))

# Symbol 到 Exchange 的映射
EXCHANGE_MAP = {
    # NASDAQ 科技股
    "AAPL": "NASDAQ", "MSFT": "NASDAQ", "GOOGL": "NASDAQ", "GOOG": "NASDAQ", "AMZN": "NASDAQ",
    "NVDA": "NASDAQ", "META": "NASDAQ", "TSLA": "NASDAQ", "AMD": "NASDAQ", "INTC": "NASDAQ",
    "NFLX": "NASDAQ", "ADBE": "NASDAQ", "CRM": "NYSE", "ORCL": "NYSE", "CSCO": "NASDAQ",
    "IBM": "NYSE", "AVGO": "NASDAQ", "QCOM": "NASDAQ", "TXN": "NASDAQ", "NOW": "NYSE",
    "SNOW": "NYSE", "ZM": "NASDAQ", "UBER": "NYSE", "LYFT": "NASDAQ", "ABNB": "NASDAQ",
    "DDOG": "NASDAQ", "MDB": "NASDAQ", "CRWD": "NASDAQ", "OKTA": "NASDAQ", "NET": "NYSE",
    "DOCU": "NASDAQ", "SQ": "NYSE", "SHOP": "NYSE", "SPOT": "NYSE", "RBLX": "NYSE",
    "TWLO": "NYSE", "FSLY": "NYSE", "PLTR": "NASDAQ", "PYPL": "NASDAQ",
    
    # 金融
    "JPM": "NYSE", "BAC": "NYSE", "WFC": "NYSE", "GS": "NYSE", "MS": "NYSE",
    "C": "NYSE", "USB": "NYSE", "PNC": "NYSE", "TFC": "NYSE", "BK": "NYSE",
    "STT": "NYSE", "BLK": "NYSE", "BX": "NYSE", "KKR": "NYSE", "APO": "NYSE",
    "CG": "NASDAQ", "MSCI": "NYSE", "SPGI": "NYSE", "MCO": "NYSE", "V": "NYSE",
    "MA": "NYSE", "AXP": "NYSE", "DFS": "NYSE", "COF": "NYSE", "SYF": "NYSE",
    "ALLY": "NYSE",
    
    # 医疗
    "JNJ": "NYSE", "UNH": "NYSE", "PFE": "NYSE", "ABBV": "NYSE", "MRK": "NYSE",
    "LLY": "NYSE", "TMO": "NYSE", "ABT": "NYSE", "BMY": "NYSE", "AMGN": "NASDAQ",
    "GILD": "NASDAQ", "REGN": "NASDAQ", "VRTX": "NASDAQ", "BIIB": "NASDAQ",
    "ISRG": "NASDAQ", "ZTS": "NYSE", "CVS": "NYSE", "CI": "NYSE", "HUM": "NYSE",
    "MDT": "NYSE", "SYK": "NYSE", "BSX": "NYSE", "EW": "NYSE", "DXCM": "NASDAQ",
    "ABMD": "NASDAQ", "ILMN": "NASDAQ", "IQV": "NYSE", "DHR": "NYSE", "A": "NYSE",
    
    # 消费
    "WMT": "NYSE", "HD": "NYSE", "COST": "NASDAQ", "NKE": "NYSE", "SBUX": "NASDAQ",
    "MCD": "NYSE", "DIS": "NYSE", "PG": "NYSE", "KO": "NYSE", "PEP": "NASDAQ",
    "TGT": "NYSE", "DG": "NYSE", "DLTR": "NASDAQ", "BBY": "NYSE", "TJX": "NYSE",
    "ROST": "NASDAQ", "BURL": "NYSE", "GPS": "NYSE", "LULU": "NASDAQ", "UA": "NYSE",
    "UAA": "NYSE", "DECK": "NYSE", "SKX": "NYSE", "CROX": "NASDAQ", "TPR": "NYSE",
    "COH": "NYSE", "EL": "NYSE", "PM": "NYSE", "MO": "NYSE", "GIS": "NYSE",
    "K": "NYSE", "CPB": "NYSE", "HSY": "NYSE", "MDLZ": "NASDAQ", "KHC": "NASDAQ",
    "SJM": "NYSE", "CAG": "NYSE", "TSN": "NYSE", "ADM": "NYSE", "BG": "NYSE",
    "INGR": "NYSE", "FLO": "NYSE",
    
    # 能源
    "XOM": "NYSE", "CVX": "NYSE", "COP": "NYSE", "EOG": "NYSE", "PXD": "NYSE",
    "MPC": "NYSE", "VLO": "NYSE", "PSX": "NYSE", "WMB": "NYSE", "OKE": "NYSE",
    "KMI": "NYSE", "EPD": "NYSE", "ET": "NYSE", "MPLX": "NYSE", "ENB": "NYSE",
    "TRP": "NYSE", "OXY": "NYSE", "DVN": "NYSE", "FANG": "NASDAQ", "MRO": "NYSE",
    "HES": "NYSE", "APA": "NASDAQ", "OVV": "NYSE", "EQT": "NYSE", "RRC": "NYSE",
    "AR": "NYSE", "SWN": "NYSE", "CTRA": "NYSE", "CHK": "NASDAQ", "MUR": "NYSE",
    
    # 工业
    "BA": "NYSE", "CAT": "NYSE", "GE": "NYSE", "RTX": "NYSE", "LMT": "NYSE",
    "NOC": "NYSE", "GD": "NYSE", "HII": "NYSE", "TDG": "NYSE", "HON": "NASDAQ",
    "MMM": "NYSE", "ITW": "NYSE", "ILL": "NYSE", "SWK": "NYSE", "SNA": "NYSE",
    "GRMN": "NYSE", "TRMB": "NASDAQ", "FLS": "NYSE", "PNR": "NYSE", "UPS": "NYSE",
    "FDX": "NYSE", "CSX": "NASDAQ", "UNP": "NYSE", "NSC": "NYSE", "KSU": "NYSE",
    "LSTR": "NASDAQ", "CHRW": "NASDAQ", "EXPD": "NASDAQ",
    
    # 通信
    "CMCSA": "NASDAQ", "VZ": "NYSE", "T": "NYSE", "TMUS": "NASDAQ", "CHTR": "NASDAQ",
    "ATVI": "NASDAQ", "EA": "NASDAQ", "TTWO": "NASDAQ", "MTCH": "NASDAQ",
    "BMBL": "NASDAQ", "PINS": "NYSE", "SNAP": "NYSE", "TWTR": "NYSE",
    
    # 房地产
    "AMT": "NYSE", "PLD": "NYSE", "CCI": "NYSE", "EQIX": "NASDAQ", "DLR": "NYSE",
    "PSA": "NYSE", "O": "NYSE", "WPC": "NYSE", "NNN": "NYSE", "STOR": "NYSE",
    "SPG": "NYSE", "MAC": "NYSE", "GGP": "NYSE", "KIM": "NYSE", "FRT": "NYSE",
    "REG": "NASDAQ", "BXP": "NYSE", "SLG": "NYSE", "VTR": "NYSE", "HCP": "NYSE",
    
    # 材料
    "LIN": "NYSE", "APD": "NYSE", "ECL": "NYSE", "SHW": "NYSE", "PPG": "NYSE",
    "DD": "NYSE", "LYB": "NYSE", "ALB": "NYSE", "SQM": "NYSE", "FMC": "NYSE",
    "NEM": "NYSE", "GOLD": "NYSE", "AEM": "NYSE", "KL": "NYSE", "AU": "NYSE",
    "ABX": "NYSE", "FCX": "NYSE", "SCCO": "NYSE", "TECK": "NYSE", "VALE": "NYSE",
    
    # 公用事业
    "NEE": "NYSE", "DUK": "NYSE", "SO": "NYSE", "D": "NYSE", "AEP": "NASDAQ",
    "EXC": "NASDAQ", "SRE": "NYSE", "XEL": "NASDAQ", "ED": "NYSE", "FE": "NYSE",
    "PEG": "NYSE", "ETR": "NYSE", "AEE": "NYSE", "CNP": "NYSE", "NI": "NYSE",
    "WEC": "NYSE", "CMS": "NYSE", "ATO": "NYSE", "LNT": "NASDAQ", "EVRG": "NYSE",
}


class TradingViewDataFetcher:
    """TradingView 数据获取器"""
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        初始化 TradingView 数据获取器
        
        Args:
            username: TradingView 用户名（可选，用于获取更多数据）
            password: TradingView 密码（可选）
        """
        self.tv = TvDatafeed(username, password)
        self.interval_map = {
            '1m': Interval.in_1_minute,
            '5m': Interval.in_5_minute,
            '15m': Interval.in_15_minute,
            '30m': Interval.in_30_minute,
            '1h': Interval.in_1_hour,
            '2h': Interval.in_2_hour,
            '4h': Interval.in_4_hour,
            '1d': Interval.in_daily,
            '1W': Interval.in_weekly,
            '1M': Interval.in_monthly,
        }
    
    def get_exchange(self, symbol: str) -> str:
        """获取 symbol 对应的交易所"""
        return EXCHANGE_MAP.get(symbol.upper(), "NASDAQ")  # 默认为 NASDAQ
    
    def fetch_ohlcv(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        interval: str = "1d",
        n_bars: int = 10000,
        retry: int = 3,
        delay: float = 1.0,
    ) -> Optional[pd.DataFrame]:
        """
        获取 OHLCV 数据
        
        Args:
            symbol: 股票代码
            exchange: 交易所代码（None 则自动检测）
            interval: 时间周期 (1m, 5m, 1h, 1d, 1W, 1M)
            n_bars: 获取的 K 线数量（最大 50000）
            retry: 重试次数
            delay: 重试延迟（秒）
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        symbol = symbol.upper()
        if exchange is None:
            exchange = self.get_exchange(symbol)
        
        tv_interval = self.interval_map.get(interval, Interval.in_daily)
        
        for attempt in range(retry):
            try:
                # 获取数据
                df = self.tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=tv_interval,
                    n_bars=min(n_bars, 50000),  # TradingView 限制
                )
                
                if df is None or df.empty:
                    if attempt < retry - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    return None
                
                # 标准化列名
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # 确保列名首字母大写
                df.columns = [col.capitalize() for col in df.columns]
                
                # 确保所有必需的列都存在
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"  Warning: Missing column {col} for {symbol}")
                        return None
                
                # 选择需要的列
                df = df[required_cols]
                
                # 确保索引是 datetime
                df.index = pd.to_datetime(df.index)
                df.index.name = 'Date'
                
                return df
                
            except Exception as e:
                print(f"  Attempt {attempt + 1}/{retry} failed: {str(e)[:50]}")
                if attempt < retry - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    return None
        
        return None
    
    def fetch_history(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        exchange: Optional[str] = None,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        获取指定日期范围的历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)，默认为今天
            exchange: 交易所代码
            interval: 时间周期
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 计算需要的 bar 数量
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if interval == "1d":
            days = (end - start).days
            n_bars = min(days + 252, 50000)  # 额外加一年数据确保覆盖
        elif interval == "1W":
            weeks = (end - start).days // 7
            n_bars = min(weeks + 52, 50000)
        elif interval == "1M":
            months = (end - start).days // 30
            n_bars = min(months + 12, 50000)
        else:
            n_bars = 50000
        
        # 获取数据
        df = self.fetch_ohlcv(symbol, exchange, interval, n_bars)
        
        if df is None or df.empty:
            return None
        
        # 过滤日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return df


def download_stock(
    fetcher: TradingViewDataFetcher,
    symbol: str,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """下载单只股票数据"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"  Fetching {symbol} from {start_date} to {end_date}...", end=" ")
    
    df = fetcher.fetch_history(symbol, start_date, end_date, interval=interval)
    
    if df is not None and len(df) > 0:
        print(f"OK ({len(df)} bars)")
        return df
    else:
        print("FAILED")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download US stock data from TradingView")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--start-date", default="1990-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1W', '1M'], help="Data interval")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--format", default="parquet", choices=['parquet', 'csv', 'feather'], help="Output format")
    parser.add_argument("--username", help="TradingView username (optional)")
    parser.add_argument("--password", help="TradingView password (optional)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    parser.add_argument("--force", action="store_true", help="Force re-download existing files")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted download")
    
    args = parser.parse_args()
    
    # 确定要下载的 symbols
    symbols = args.symbols if args.symbols else US_STOCKS
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查已存在的文件
    existing_files = set()
    if not args.force:
        for fmt in ['parquet', 'csv', 'feather']:
            existing_files.update(f.stem for f in args.output_dir.glob(f"*.{fmt}"))
    
    if args.resume:
        symbols = [s for s in symbols if s not in existing_files]
    
    print(f"=" * 60)
    print(f"TradingView US Stock Data Downloader")
    print(f"=" * 60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Existing files: {len(existing_files)}")
    print(f"To download: {len(symbols)}")
    print(f"Date range: {args.start_date} to {args.end_date or 'today'}")
    print(f"Interval: {args.interval}")
    print(f"Output format: {args.format}")
    print(f"Output directory: {args.output_dir}")
    print(f"=" * 60)
    
    # 初始化 fetcher
    fetcher = TradingViewDataFetcher(args.username, args.password)
    
    # 下载数据
    success = 0
    failed = []
    skipped = 0
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        # 检查是否已存在
        if symbol in existing_files and not args.force:
            print(f"[{i+1}/{len(symbols)}] {symbol}: SKIP (exists)")
            skipped += 1
            continue
        
        print(f"[{i+1}/{len(symbols)}] {symbol}:", end=" ")
        
        df = download_stock(
            fetcher,
            symbol,
            args.start_date,
            args.end_date,
            args.interval
        )
        
        if df is not None and len(df) > 100:
            # 保存文件
            output_path = args.output_dir / f"{symbol}.{args.format}"
            
            if args.format == "parquet":
                df.to_parquet(output_path)
            elif args.format == "csv":
                df.to_csv(output_path)
            elif args.format == "feather":
                df.to_feather(output_path)
            
            success += 1
        else:
            failed.append(symbol)
        
        # 延迟避免请求过快
        if args.delay > 0 and i < len(symbols) - 1:
            time.sleep(args.delay)
        
        # 每10个显示进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(symbols) - i - 1) / rate if rate > 0 else 0
            print(f"  -> Progress: {i+1}/{len(symbols)} | Rate: {rate:.1f} stocks/sec | ETA: {remaining/60:.1f} min")
    
    # 总结
    elapsed = time.time() - start_time
    print(f"=" * 60)
    print(f"Download Summary")
    print(f"=" * 60)
    print(f"Success: {success}/{len(symbols)}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Rate: {len(symbols)/elapsed:.1f} stocks/sec")
    
    if failed:
        print(f"\nFailed symbols: {failed}")
        # 保存失败列表
        failed_file = args.output_dir / "failed_symbols.txt"
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed))
        print(f"Failed list saved to: {failed_file}")


if __name__ == "__main__":
    main()
