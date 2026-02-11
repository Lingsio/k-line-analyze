"""
Download Chinese A-share stock data using baostock.
More reliable than akshare for batch downloads.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Try to import baostock
try:
    import baostock as bs
except ImportError:
    print("Installing baostock...")
    os.system(f"{sys.executable} -m pip install baostock -q")
    import baostock as bs

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "cn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Top 50 Chinese A-share stocks (format: sh/sz + code)
CN_STOCKS = [
    # 金融
    ("sh.600519", "贵州茅台"),
    ("sh.601318", "中国平安"),
    ("sh.600036", "招商银行"),
    ("sh.601166", "兴业银行"),
    ("sh.600000", "浦发银行"),
    ("sh.601288", "农业银行"),
    ("sh.601398", "工商银行"),
    ("sh.601939", "建设银行"),
    ("sh.601988", "中国银行"),
    ("sh.600030", "中信证券"),

    # 科技/互联网
    ("sz.000063", "中兴通讯"),
    ("sz.002415", "海康威视"),
    ("sz.300750", "宁德时代"),
    ("sz.002594", "比亚迪"),
    ("sz.300059", "东方财富"),
    ("sz.002475", "立讯精密"),
    ("sz.300124", "汇川技术"),
    ("sh.688981", "中芯国际"),
    ("sz.002230", "科大讯飞"),
    ("sz.300760", "迈瑞医疗"),

    # 消费
    ("sz.000858", "五粮液"),
    ("sz.000568", "泸州老窖"),
    ("sh.600887", "伊利股份"),
    ("sz.002714", "牧原股份"),
    ("sh.603288", "海天味业"),
    ("sz.000333", "美的集团"),
    ("sz.000651", "格力电器"),
    ("sh.600690", "海尔智家"),
    ("sz.002352", "顺丰控股"),
    ("sh.601888", "中国中免"),

    # 医药
    ("sh.600276", "恒瑞医药"),
    ("sz.000538", "云南白药"),
    ("sh.600196", "复星医药"),
    ("sz.002007", "华兰生物"),
    ("sz.300122", "智飞生物"),

    # 新能源
    ("sh.601012", "隆基绿能"),
    ("sz.002459", "晶澳科技"),
    ("sh.600438", "通威股份"),
    ("sz.002129", "TCL中环"),
    ("sz.300274", "阳光电源"),

    # 工业/材料
    ("sh.600309", "万华化学"),
    ("sh.601899", "紫金矿业"),
    ("sh.600585", "海螺水泥"),
    ("sh.601668", "中国建筑"),
    ("sh.600031", "三一重工"),

    # 其他蓝筹
    ("sh.601857", "中国石油"),
    ("sh.600028", "中国石化"),
    ("sh.601088", "中国神华"),
    ("sh.600900", "长江电力"),
    ("sz.000001", "平安银行"),
]

def download_stock(bs_code, name, start_date="2015-01-01", end_date=None):
    """Download a single stock's daily data using baostock."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )

        if rs.error_code != '0':
            print(f"  Error: {rs.error_msg}", flush=True)
            return None

        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())

        if len(data_list) == 0:
            print(f"  No data", flush=True)
            return None

        df = pd.DataFrame(data_list, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Convert types
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.set_index('Date')
        df = df.sort_index()
        df = df.dropna()

        return df

    except Exception as e:
        print(f"  Exception: {str(e)[:50]}", flush=True)
        return None

def main():
    print(f"Downloading {len(CN_STOCKS)} Chinese stocks using baostock...", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print("-" * 50, flush=True)

    # Login to baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"Login failed: {lg.error_msg}", flush=True)
        return
    print("Logged in to baostock", flush=True)

    success = 0
    failed = []

    for i, (bs_code, name) in enumerate(CN_STOCKS):
        # Extract simple code (e.g., sh.600519 -> 600519)
        simple_code = bs_code.split('.')[1]
        output_path = OUTPUT_DIR / f"{simple_code}.parquet"

        # Skip if already exists
        if output_path.exists():
            print(f"[{i+1}/{len(CN_STOCKS)}] {simple_code} - EXISTS, skipping", flush=True)
            success += 1
            continue

        print(f"[{i+1}/{len(CN_STOCKS)}] {simple_code}...", end=" ", flush=True)

        df = download_stock(bs_code, name)

        if df is not None and len(df) > 100:
            df.to_parquet(output_path)
            print(f"OK ({len(df)} days)", flush=True)
            success += 1
        else:
            failed.append(simple_code)
            print("SKIP", flush=True)

    # Logout
    bs.logout()

    print("-" * 50, flush=True)
    print(f"Total: {success}/{len(CN_STOCKS)}", flush=True)
    if failed:
        print(f"Failed: {failed}", flush=True)

if __name__ == "__main__":
    main()
