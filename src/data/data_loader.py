import yfinance as yf
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

# List of 100 Representative A-share stocks (covering various sectors)
# Mix of Large Caps (CSI300) and some Volatile stocks for pattern variety.
# Codes need .SS (Shanghai) or .SZ (Shenzhen) suffix for yfinance.
TICKERS = [
    "600519.SS", "601398.SS", "601288.SS", "601988.SS", "601857.SS", "600036.SS", "601318.SS", "600900.SS", "601012.SS", "600276.SS",
    "603288.SS", "600887.SS", "600030.SS", "600000.SS", "601328.SS", "600028.SS", "600048.SS", "601166.SS", "601668.SS", "601628.SS",
    "000858.SZ", "000333.SZ", "002415.SZ", "002594.SZ", "300750.SZ", "300760.SZ", "000651.SZ", "000001.SZ", "000002.SZ", "002714.SZ",
    "600019.SS", "601998.SS", "601390.SS", "601818.SS", "601088.SS", "600016.SS", "600104.SS", "600018.SS", "600050.SS", "601601.SS",
    "601138.SS", "600585.SS", "600690.SS", "600837.SS", "600031.SS", "600009.SS", "600309.SS", "600547.SS", "600048.SS", "601111.SS",
    "000725.SZ", "002475.SZ", "002142.SZ", "000063.SZ", "000100.SZ", "000776.SZ", "000786.SZ", "002027.SZ", "002241.SZ", "002304.SZ",
    "300015.SZ", "300059.SZ", "300122.SZ", "300014.SZ", "300124.SZ", "300274.SZ", "300142.SZ", "300144.SZ", "300296.SZ", "300601.SZ",
    "600340.SS", "600176.SS", "600588.SS", "600741.SS", "600570.SS", "600703.SS", "600233.SS", "600346.SS", "600893.SS", "601155.SS",
    "002001.SZ", "002007.SZ", "002044.SZ", "002050.SZ", "002129.SZ", "002153.SZ", "002179.SZ", "002230.SZ", "002236.SZ", "002250.SZ",
    "300001.SZ", "300002.SZ", "300003.SZ", "300009.SZ", "300010.SZ", "300012.SZ", "300017.SZ", "300024.SZ", "300033.SZ", "600111.SS"
]

# Top 100 US Stocks (Approximate S&P 100 / Large Cap Tech & Finance)
US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY",
    "AVGO", "JPM", "V", "XOM", "UNH", "WMT", "MA", "JNJ", "PG", "HD",
    "COST", "MRK", "ORCL", "ABBV", "CVX", "BAC", "CRM", "AMD", "PEP", "KO",
    "NFLX", "TMO", "LIN", "WFC", "DIS", "ADBE", "MCD", "CSCO", "ABT", "TMUS",
    "QCOM", "INTU", "VZ", "AMAT", "IBM", "CMCSA", "PFE", "INTC", "UBER", "HON",
    "DHR", "UNP", "TXN", "AMGN", "LOW", "PM", "SPGI", "CAT", "AXP", "GS",
    "GE", "RTX", "ISRG", "BLK", "SYK", "T", "PGR", "BKNG", "NEE", "ELV",
    "LRCX", "VRTX", "TJX", "MS", "MDT", "GILD", "DE", "BSX", "ADP", "MMC",
    "CI", "ADI", "MDLZ", "LMT", "PANW", "BX", "BA", "PLD", "FI", "CB",
    "ETN", "BMY", "KLAC", "SNPS", "REGN", "C", "SHW", "VRT", "ZTS", "SBUX"
]

ALL_TICKERS = TICKERS + US_TICKERS

def download_ticker(ticker):
    try:
        print(f"Downloading {ticker}...")
        # Get data from 1990 to present to maximize history
        df = yf.download(ticker, start="1990-01-01", end=datetime.today().strftime('%Y-%m-%d'), progress=False)
        
        if len(df) > 0:
            # Flatten MultiIndex columns if present (yfinance update issue)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Save to CSV
            save_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            df.to_csv(save_path)
            return True
        else:
            print(f"Warning: No data for {ticker}")
            return False
            
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return False

def main():
    print(f"Starting download for {len(TICKERS)} stocks...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(download_ticker, ALL_TICKERS)
        success_count = sum(results)
        
    print(f"\nDownload complete. Successfully downloaded {success_count}/{len(ALL_TICKERS)} stocks.")
    print(f"Data saved to {DATA_DIR}")

if __name__ == "__main__":
    main()
