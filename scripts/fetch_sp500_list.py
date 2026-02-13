"""
Fetch S&P 500 constituents list from Wikipedia or other sources.
Updates the tradingview_fetcher.py with the complete list.
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_sp500_from_wikipedia():
    """Fetch S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        # Read tables from Wikipedia
        tables = pd.read_html(url)
        
        # The first table contains the S&P 500 constituents
        df = tables[0]
        
        # Extract symbols
        symbols = df['Symbol'].tolist()
        
        # Get exchange info from the table
        # Wikipedia doesn't always have exchange, but we can infer
        return symbols, df
        
    except Exception as e:
        print(f"Error fetching from Wikipedia: {e}")
        return None, None


def fetch_sp500_from_github():
    """Fetch S&P 500 constituents from GitHub dataset."""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        symbols = df['Symbol'].tolist()
        
        return symbols, df
        
    except Exception as e:
        print(f"Error fetching from GitHub: {e}")
        return None, None


def get_exchange_from_sector(symbol, sector, sub_industry):
    """Infer exchange based on sector and known patterns."""
    # NASDAQ-heavy sectors
    nasdaq_sectors = [
        'Information Technology',
        'Communication Services',
        'Consumer Discretionary'
    ]
    
    # Known NASDAQ stocks
    known_nasdaq = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ADBE',
        'COST', 'NFLX', 'AMD', 'PEP', 'CSCO', 'TMUS', 'INTC', 'QCOM', 'TXN', 'AMGN',
        'HON', 'SBUX', 'INTU', 'BKNG', 'LRCX', 'GILD', 'ADP', 'VRTX', 'MDLZ', 'ISRG',
        'REGN', 'PYPL', 'ABNB', 'PANW', 'MU', 'ADI', 'KLAC', 'SNPS', 'CDNS', 'MAR',
        'CSX', 'MELI', 'ABNB', 'LULU', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'FTNT',
        'OKTA', 'ZM', 'DOCU', 'ROKU', 'UBER', 'LYFT', 'SHOP', 'SQ', 'SPOT', 'SNOW',
        'CRWD', 'DDOG', 'NET', 'FSLY', 'PLTR', 'U', 'RBLX', 'HOOD', 'COIN', 'AFRM',
        'SOFI', 'LCID', 'RIVN', 'NIO', 'XPEV', 'LI', 'DASH', 'TOST', 'BILL', 'ASAN',
        'MDB', 'ESTC', 'SPLK', 'NOW', 'VEEV', 'TWLO', 'PLAN', 'SMAR', 'HUBS', 'DDOG',
        'NET', 'FSLY', 'CFLT', 'SNOW', 'MDB', 'S', 'CRWD', 'PANW', 'CYBR', 'QLYS',
        'FTNT', 'CHKP', 'OKTA', 'DUOL', 'WIX', 'APPN', 'PLTR', 'AI', 'PATH', 'U',
        'RBLX', 'U', 'EA', 'ATVI', 'TTWO', 'ZNGA', 'RBLX', 'U', 'MTCH', 'BMBL',
        'OKC', 'IAC', 'MTCH', 'BMBL', 'SQ', 'PYPL', 'AFRM', 'SOFI', 'UPST', 'LMND',
        'ROOT', 'HCP', 'VEEV', 'TDOC', 'AMWL', 'PGNY', 'NVAX', 'MRNA', 'BNTX', 'VIR',
        'ALNY', 'IONS', 'SRPT', 'BMRN', 'VRTX', 'REGN', 'INCY', 'BIIB', 'GILD', 'AMGN',
        'ILMN', 'PACB', 'TWST', 'DNA', 'TDOC', 'AMWL', 'HIMS', 'OSH', 'AGL', 'LFST',
        'BHVN', 'CERE', 'SANA', 'NTLA', 'EDIT', 'BEAM', 'CRSP', 'VCEL', 'BLUE', 'FATE',
        'PSNL', 'SLGC', 'APPS', 'TTD', 'MGNI', 'PUBM', 'DV', 'TRMR', 'CRTO', 'QUOT',
        'TBLA', 'IAS', 'MAX', 'PERI', 'ZETA', 'FORA', 'DJCO', 'NXST', 'TRCO', 'GTN',
        'SBGI', 'FOXA', 'FOX', 'NWSA', 'NWS', 'NYT', 'GCI', 'LEE', 'SSP', 'MDP',
        'TGNA', 'GTN', 'SBGI', 'NXST', 'FOX', 'FOXA', 'NWS', 'NWSA', 'DIS', 'WBD',
        'PARA', 'LYV', 'SIRI', 'P', 'SPOT', 'SXM', 'AUD', 'IHRT', 'ETSY', 'CHWY',
        'CVNA', 'VRM', 'KMX', 'AN', 'LAD', 'ABG', 'SAH', 'CPRT', 'IAA', 'KAR',
        'TRUE', 'VRM', 'CVNA', 'OPEN', 'RDFN', 'Z', 'ZG', 'EXPI', 'COMP', 'LE',
        'CONN', 'BBBY', 'PRTS', 'KSS', 'JWN', 'M', 'DDS', 'BURL', 'ROST', 'TJX',
        'TJX', 'ROST', 'BURL', 'AEO', 'ANF', 'GPS', 'JWN', 'KSS', 'M', 'DDS',
        'PLCE', 'CRI', 'GIL', 'HBI', 'GIII', 'RL', 'TPR', 'CPRI', 'COH', 'EL',
        'ULTA', 'BBWI', 'TGT', 'DG', 'DLTR', 'FIVE', 'BIG', 'OLLI', 'COST', 'WMT',
        'COST', 'WMT', 'TGT', 'DG', 'DLTR', 'FIVE', 'BIG', 'OLLI', 'PSMT', 'IMKTA',
        'VLGEA', 'UNFI', 'SFM', 'WMK', 'GO', 'NGVC', 'IMKTA', 'PSMT', 'SFM', 'GO',
        'NGVC', 'IMKTA', 'PSMT', 'SFM', 'GO', 'NGVC', 'WFM', 'SFM', 'UNFI', 'ANDE',
        'BG', 'CHEF', 'CORE', 'DIT', 'NAII', 'SPTN', 'USFD', 'HFFG', 'JBSS', 'LANC',
        'MKC', 'SJM', 'CPB', 'CAG', 'GIS', 'K', 'HSY', 'HRL', 'KHC', 'MDLZ',
        'MKC', 'SJM', 'CPB', 'CAG', 'GIS', 'K', 'HSY', 'HRL', 'KHC', 'MDLZ',
        'PEP', 'KO', 'MNST', 'FIZZ', 'CCEP', 'KDP', 'MNST', 'FIZZ', 'CELH', 'OTLY',
        'BYND', 'TTCF', 'HRL', 'TSN', 'PPC', 'SAFM', 'SEB', 'CALM', 'POST', 'KHC',
        'K', 'CPB', 'GIS', 'CAG', 'SJM', 'HSY', 'HRL', 'MDLZ', 'MKC', 'PEP',
        'KO', 'MNST', 'FIZZ', 'CCEP', 'KDP', 'MNST', 'FIZZ', 'CELH', 'OTLY', 'BYND',
        'TTCF', 'HRL', 'TSN', 'PPC', 'SAFM', 'SEB', 'CALM', 'POST', 'KHC', 'K',
        'CPB', 'GIS', 'CAG', 'SJM', 'HSY', 'HRL', 'MDLZ', 'MKC', 'PEP', 'KO',
        'MNST', 'FIZZ', 'CCEP', 'KDP', 'MNST', 'FIZZ', 'CELH', 'OTLY', 'BYND', 'TTCF'
    }
    
    if symbol in known_nasdaq:
        return 'NASDAQ'
    
    if sector in nasdaq_sectors:
        return 'NASDAQ'
    
    # Default to NYSE
    return 'NYSE'


def update_tradingview_fetcher(symbols, df_info=None):
    """Update the tradingview_fetcher.py file with new symbols."""
    
    tv_file = Path(__file__).parent.parent / "core" / "tradingview_fetcher.py"
    
    # Read current content
    with open(tv_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Generate new exchange map
    exchange_map_lines = ["# Symbol 到 Exchange 的映射表", "EXCHANGE_MAP = {"]
    
    for symbol in sorted(set(symbols)):
        symbol = symbol.strip()
        if not symbol:
            continue
            
        # Try to get sector info if available
        sector = None
        sub_industry = None
        if df_info is not None:
            row = df_info[df_info['Symbol'] == symbol]
            if not row.empty:
                sector = row.iloc[0].get('GICS Sector') or row.iloc[0].get('Sector')
                sub_industry = row.iloc[0].get('GICS Sub-Industry') or row.iloc[0].get('Sub-Industry')
        
        exchange = get_exchange_from_sector(symbol, sector, sub_industry)
        exchange_map_lines.append(f'    "{symbol}": "{exchange}",')
    
    exchange_map_lines.append("}")
    exchange_map_text = '\n'.join(exchange_map_lines)
    
    # Generate new DEFAULT_US_STOCKS
    stocks_list_lines = ["# S&P 500 成分股列表（共503只）", "DEFAULT_US_STOCKS = ["]
    
    for i, symbol in enumerate(symbols):
        if i % 10 == 0:
            stocks_list_lines.append("")
            stocks_list_lines.append(f"    # Row {i//10 + 1}")
        stocks_list_lines.append(f'    "{symbol}",')
    
    stocks_list_lines.append("]")
    stocks_list_text = '\n'.join(stocks_list_lines)
    
    # Replace EXCHANGE_MAP in content
    import re
    
    # Find and replace EXCHANGE_MAP
    exchange_pattern = r'# Symbol 到 Exchange 的映射表.*?EXCHANGE_MAP = \{.*?\}'
    if re.search(exchange_pattern, content, re.DOTALL):
        content = re.sub(exchange_pattern, exchange_map_text, content, flags=re.DOTALL)
    
    # Find and replace DEFAULT_US_STOCKS
    stocks_pattern = r'# (常用美股列表|S&P 500 成分股列表).*?DEFAULT_US_STOCKS = \[.*?\]'
    if re.search(stocks_pattern, content, re.DOTALL):
        content = re.sub(stocks_pattern, stocks_list_text, content, flags=re.DOTALL)
    
    # Write back
    with open(tv_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated {tv_file}")
    print(f"  - Exchange map: {len(set(symbols))} symbols")
    print(f"  - Stock list: {len(symbols)} symbols")


def main():
    print("Fetching S&P 500 constituents...")
    
    # Try Wikipedia first
    symbols, df = fetch_sp500_from_wikipedia()
    
    if symbols is None:
        print("Wikipedia failed, trying GitHub...")
        symbols, df = fetch_sp500_from_github()
    
    if symbols is None:
        print("Failed to fetch S&P 500 list")
        return
    
    print(f"\nFound {len(symbols)} symbols")
    print(f"\nFirst 10: {symbols[:10]}")
    print(f"Last 10: {symbols[-10:]}")
    
    # Update tradingview_fetcher.py
    print("\nUpdating tradingview_fetcher.py...")
    update_tradingview_fetcher(symbols, df)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
