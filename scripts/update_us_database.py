"""
美股本地数据库管理脚本

功能：
- 初始化/更新本地美股数据库
- 支持增量更新
- 支持全量重新下载
- 显示数据库统计信息

使用方法：
    python update_us_database.py --update          # 增量更新所有股票
    python update_us_database.py --update --force  # 强制重新下载
    python update_us_database.py --info            # 显示数据库信息
    python update_us_database.py --symbol AAPL     # 更新特定股票
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tradingview_fetcher import USStockDatabase, DEFAULT_US_STOCKS, TradingViewFetcher


def main():
    parser = argparse.ArgumentParser(
        description="US Stock Database Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 初始化/更新所有股票（增量）
  python update_us_database.py --update
  
  # 强制重新下载所有股票
  python update_us_database.py --update --force
  
  # 只更新特定股票
  python update_us_database.py --symbol AAPL MSFT GOOGL
  
  # 显示数据库统计信息
  python update_us_database.py --info
  
  # 列出所有本地股票
  python update_us_database.py --list
  
  # 查看特定股票信息
  python update_us_database.py --show AAPL
        """
    )
    
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="数据存储目录（默认: data/raw/us）")
    parser.add_argument("--update", action="store_true",
                        help="更新数据库")
    parser.add_argument("--force", action="store_true",
                        help="强制重新下载（忽略本地缓存）")
    parser.add_argument("--symbol", nargs="+",
                        help="指定股票代码")
    parser.add_argument("--start-date", default="1990-01-01",
                        help="数据开始日期")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="请求间隔（秒）")
    parser.add_argument("--info", action="store_true",
                        help="显示数据库统计信息")
    parser.add_argument("--list", action="store_true",
                        help="列出所有本地股票")
    parser.add_argument("--show", metavar="SYMBOL",
                        help="显示特定股票的数据预览")
    parser.add_argument("--export", metavar="PATH",
                        help="导出股票数据到CSV")
    parser.add_argument("--username",
                        help="TradingView 用户名（可选）")
    parser.add_argument("--password",
                        help="TradingView 密码（可选）")
    
    args = parser.parse_args()
    
    # 初始化数据库
    if args.data_dir is None:
        args.data_dir = Path(__file__).parent.parent / "data" / "raw" / "us"
    
    db = USStockDatabase(
        data_dir=args.data_dir,
        username=args.username,
        password=args.password,
    )
    
    # 显示数据库信息
    if args.info:
        print("=" * 70)
        print("US Stock Database Information")
        print("=" * 70)
        print(f"Data directory: {db.data_dir}")
        print(f"Total stocks: {len(db.list_stocks())}")
        
        if db.metadata:
            df_summary = db.get_database_summary()
            if not df_summary.empty:
                print("\n--- Summary Statistics ---")
                print(f"Total rows: {df_summary['rows'].sum():,}")
                print(f"Average rows per stock: {df_summary['rows'].mean():.0f}")
                print(f"\nOldest data: {df_summary['start_date'].min()}")
                print(f"Newest data: {df_summary['end_date'].max()}")
                print(f"\n--- Top 10 by data size ---")
                top10 = df_summary.nlargest(10, 'rows')[['symbol', 'rows', 'end_date']]
                print(top10.to_string(index=False))
        
        print("=" * 70)
        return
    
    # 列出所有股票
    if args.list:
        stocks = db.list_stocks()
        print(f"Total stocks in database: {len(stocks)}")
        print("\n".join(stocks))
        return
    
    # 显示特定股票
    if args.show:
        symbol = args.show.upper()
        df = db.load_stock(symbol)
        if df is not None:
            print(f"\n{symbol} Data Preview:")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nLast 5 rows:")
            print(df.tail())
            print("\nStatistics:")
            print(df.describe())
        else:
            print(f"Symbol {symbol} not found in database")
        return
    
    # 导出数据
    if args.export:
        symbol = args.export.upper()
        df = db.load_stock(symbol)
        if df is not None:
            output_path = Path(f"{symbol}.csv")
            df.to_csv(output_path)
            print(f"Exported {symbol} to {output_path}")
        else:
            print(f"Symbol {symbol} not found in database")
        return
    
    # 更新数据库
    if args.update:
        symbols = args.symbol if args.symbol else DEFAULT_US_STOCKS
        db.update_all(
            symbols=symbols,
            start_date=args.start_date,
            force=args.force,
            delay=args.delay,
        )
        return
    
    # 如果没有指定操作，显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
