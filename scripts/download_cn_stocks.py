"""
Download Chinese A-share stock data using akshare.
Downloads top 50 stocks by market cap.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Try to import akshare
try:
    import akshare as ak
except ImportError:
    print("Installing akshare...")
    os.system("pip install akshare -q")
    import akshare as ak

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "cn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Top 50 Chinese A-share stocks (by market cap / popularity)
CN_STOCKS = [
    # 金融
    "600519",  # 贵州茅台
    "601318",  # 中国平安
    "600036",  # 招商银行
    "601166",  # 兴业银行
    "600000",  # 浦发银行
    "601288",  # 农业银行
    "601398",  # 工商银行
    "601939",  # 建设银行
    "601988",  # 中国银行
    "600030",  # 中信证券

    # 科技/互联网
    "000063",  # 中兴通讯
    "002415",  # 海康威视
    "300750",  # 宁德时代
    "002594",  # 比亚迪
    "300059",  # 东方财富
    "002475",  # 立讯精密
    "300124",  # 汇川技术
    "688981",  # 中芯国际
    "002230",  # 科大讯飞
    "300760",  # 迈瑞医疗

    # 消费
    "000858",  # 五粮液
    "000568",  # 泸州老窖
    "600887",  # 伊利股份
    "002714",  # 牧原股份
    "603288",  # 海天味业
    "000333",  # 美的集团
    "000651",  # 格力电器
    "600690",  # 海尔智家
    "002352",  # 顺丰控股
    "601888",  # 中国中免

    # 医药
    "600276",  # 恒瑞医药
    "000538",  # 云南白药
    "600196",  # 复星医药
    "002007",  # 华兰生物
    "300122",  # 智飞生物

    # 新能源
    "601012",  # 隆基绿能
    "002459",  # 晶澳科技
    "600438",  # 通威股份
    "002129",  # TCL中环
    "300274",  # 阳光电源

    # 工业/材料
    "600309",  # 万华化学
    "601899",  # 紫金矿业
    "600585",  # 海螺水泥
    "601668",  # 中国建筑
    "600031",  # 三一重工

    # 其他蓝筹
    "601857",  # 中国石油
    "600028",  # 中国石化
    "601088",  # 中国神华
    "600900",  # 长江电力
    "000001",  # 平安银行
]

def download_stock(symbol, start_date="20150101", end_date=None):
    """Download a single stock's daily data."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    try:
        # Try A-share daily data
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )

        if df is None or len(df) == 0:
            print(f"  {symbol}: No data")
            return None

        # Rename columns to standard format
        df = df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume',
        })

        # Select and order columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()

        return df

    except Exception as e:
        print(f"  {symbol}: Error - {str(e)[:50]}")
        return None

def main():
    print(f"Downloading {len(CN_STOCKS)} Chinese stocks...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    success = 0
    failed = []

    for i, symbol in enumerate(CN_STOCKS):
        print(f"[{i+1}/{len(CN_STOCKS)}] Downloading {symbol}...", end=" ")

        df = download_stock(symbol)

        if df is not None and len(df) > 100:  # At least 100 days of data
            output_path = OUTPUT_DIR / f"{symbol}.parquet"
            df.to_parquet(output_path)
            print(f"OK ({len(df)} days)")
            success += 1
        else:
            failed.append(symbol)
            print("SKIP")

    print("-" * 50)
    print(f"Downloaded: {success}/{len(CN_STOCKS)}")
    if failed:
        print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
