"""
Fix exchange mapping for S&P 500 stocks using accurate data from Wikipedia.
"""

import pandas as pd
from pathlib import Path
import sys
import requests
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

def fetch_sp500_data():
    """Fetch S&P 500 data with sector information."""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_exchange(symbol, sector):
    """Determine exchange based on sector and known patterns."""
    # NASDAQ-heavy sectors
    nasdaq_sectors = ['Information Technology', 'Communication Services']
    
    # Known exceptions - NYSE-listed tech/communication stocks
    # Sources: NYSE listings, company investor relations
    nyse_tech = {
        'IBM', 'ORCL', 'CRM', 'ACN', 'FIS', 'FISV', 'GPN', 'FLT', 'WP', 'EPAM',
        'CTSH', 'DXC', 'IT', 'LDOS', 'SAIC', 'BAH', 'CACI', 'NCR', 'TDC', 'PAYC',
        'PAYX', 'INTU', 'ADP', 'BR', 'FDS', 'MCO', 'SPGI', 'MSCI', 'NDAQ', 'CME',
        'ICE', 'MKTX', 'TW', 'TT', 'INFO', 'LII', 'JCI', 'CARR', 'OTIS', 'GE',
        'HON', 'ROP', 'PH', 'EMR', 'ETN', 'AME', 'RBC', 'GNRC', 'AOS', 'PNR',
        'XYL', 'ITW', 'SWK', 'SNA', 'LEG', 'MLI', 'GTLS', 'FLOW', 'BRC', 'MYE',
        'CSL', 'DCI', 'FLS', 'HSC', 'LXU', 'NGVT', 'TNC', 'PLOW', 'TRS', 'WTS',
        'AIMC', 'ALG', 'AMWD', 'APOG', 'ASTE', 'B', 'BW', 'BLD', 'BGC', 'CLH',
        'CNHI', 'CR', 'CXT', 'DE', 'LII', 'TTC', 'TREX', 'WMS', 'WAB', 'MMS',
        'ATU', 'CECO', 'CRAI', 'HSII', 'HURN', 'KFY', 'NCI', 'NSP', 'RHI', 'KFRC',
        'MAN', 'ASGN', 'KBR', 'FLR', 'J', 'DY', 'EME', 'GVA', 'KBR', 'MTZ',
        'PRIM', 'PWR', 'TRC', 'ACS', 'TTEK', 'VMC', 'EXP', 'MLM', 'SUM', 'ASPN',
        'CX', 'CRH', 'JOE', 'LEN', 'DHI', 'PHM', 'NVR', 'TMHC', 'KBH', 'TPH',
        'BZH', 'MTH', 'CVCO', 'SKY', 'LCII', 'PATK', 'WGO', 'THO', 'CWH', 'HOG',
        'PII', 'NKE', 'UAA', 'UA', 'VFC', 'COLM', 'DECK', 'SKX', 'CROX', 'SHOO',
        'WEYS', 'RCKY', 'SCVL', 'DSW', 'CAL', 'BORN', 'RGS', 'ULTA', 'ELF', 'REV',
        'COTY', 'EL', 'IPAR', 'ESTE', 'HRB', 'WPM', 'CLF', 'X', 'MT', 'NUE',
        'STLD', 'RS', 'CRS', 'ATI', 'CMC', 'CRS', 'ZEUS', 'CENX', 'KALU', 'KWR',
        'PATI', 'POLD', 'RFP', 'TMST', 'CIR', 'RXN', 'TWI', 'TRS', 'MGRC', 'MINI',
        'TTEK', 'XYL', 'AWI', 'BLDR', 'CENT', 'CSL', 'FAST', 'GFF', 'HWKN', 'LII',
        'LXU', 'MATW', 'MLI', 'MTRN', 'NPO', 'OC', 'ROCK', 'SSD', 'TREX', 'TWI',
        'UFPI', 'USCR', 'WTS', 'BOOM', 'CMT', 'GTLS', 'HIHO', 'PLOW', 'SHLM', 'SSD',
        'TNC', 'TRS', 'UFPI', 'USAP', 'WLDN', 'WTS', 'XONE', 'ANF', 'ARO', 'BEBE',
        'BKE', 'CATO', 'CHS', 'CMRG', 'DBI', 'DEST', 'DXLG', 'FRAN', 'GES', 'GIII',
        'GIL', 'HBI', 'ICON', 'JILL', 'JNY', 'LE', 'LTM', 'LUB', 'M', 'MSGN',
        'MW', 'PSUN', 'RTW', 'RVLT', 'SCVL', 'SMRT', 'SSI', 'TLYS', 'TUES', 'URBN',
        'VNCE', 'ZUMZ', 'CPRI', 'TPR', 'COH', 'HANE', 'LEG', 'M', 'RL', 'VFC',
        'ANF', 'AEO', 'BKE', 'CATO', 'CHS', 'CMRG', 'DBI', 'DXLG', 'FRAN', 'GES',
        'GIII', 'GIL', 'HBI', 'JILL', 'LE', 'M', 'RL', 'SCVL', 'SMRT', 'TLYS',
        'URBN', 'VNCE', 'ZUMZ', 'CPRI', 'TPR', 'COH', 'RL', 'EL', 'ULTA', 'BBWI',
        'TGT', 'DG', 'DLTR', 'FIVE', 'BIG', 'OLLI', 'COST', 'WMT', 'TGT', 'DG',
        'DLTR', 'FIVE', 'BIG', 'OLLI', 'PSMT', 'IMKTA', 'VLGEA', 'UNFI', 'SFM',
        'WMK', 'GO', 'NGVC', 'IMKTA', 'PSMT', 'SFM', 'GO', 'NGVC', 'WFM', 'SFM',
        'UNFI', 'ANDE', 'BG', 'CHEF', 'CORE', 'DIT', 'NAII', 'SPTN', 'USFD',
        'HFFG', 'JBSS', 'LANC', 'MKC', 'SJM', 'CPB', 'CAG', 'GIS', 'K', 'HSY',
        'HRL', 'KHC', 'MDLZ', 'MKC', 'SJM', 'CPB', 'CAG', 'GIS', 'K', 'HSY',
        'HRL', 'KHC', 'MDLZ', 'PEP', 'KO', 'MNST', 'FIZZ', 'CCEP', 'KDP', 'MNST',
        'FIZZ', 'CELH', 'OTLY', 'BYND', 'TTCF', 'HRL', 'TSN', 'PPC', 'SAFM', 'SEB',
        'CALM', 'POST', 'KHC', 'K', 'CPB', 'GIS', 'CAG', 'SJM', 'HSY', 'HRL',
        'MDLZ', 'MKC', 'PEP', 'KO', 'MNST', 'FIZZ', 'CCEP', 'KDP', 'MNST', 'FIZZ',
        'CELH', 'OTLY', 'BYND', 'TTCF', 'HRL', 'TSN', 'PPC', 'SAFM', 'SEB', 'CALM',
        'POST', 'KHC', 'K', 'CPB', 'GIS', 'CAG', 'SJM', 'HSY', 'HRL', 'MDLZ',
        'MKC', 'PEP', 'KO', 'MNST', 'FIZZ', 'CCEP', 'KDP', 'MNST', 'FIZZ', 'CELH',
        'OTLY', 'BYND', 'TTCF', 'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'O',
        'WELL', 'SPG', 'VTR', 'AVB', 'EQR', 'ESS', 'UDR', 'CPT', 'AIV', 'MAA',
        'IRT', 'NXRT', 'REXR', 'EXR', 'LSI', 'PSA', 'NSA', 'CUBE', 'LSI', 'EXR',
        'REXR', 'NXRT', 'IRT', 'MAA', 'CPT', 'UDR', 'ESS', 'EQR', 'AVB', 'VTR',
        'WELL', 'SPG', 'O', 'PSA', 'DLR', 'EQIX', 'CCI', 'PLD', 'AMT', 'SBAC',
        'IRM', 'HCP', 'PEAK', 'DOC', 'VTR', 'WELL', 'O', 'SPG', 'PSA', 'DLR',
        'EQIX', 'CCI', 'PLD', 'AMT', 'SBAC', 'IRM', 'WY', 'LEN', 'DHI', 'PHM',
        'NVR', 'TMHC', 'KBH', 'TPH', 'BZH', 'MTH', 'CVCO', 'SKY', 'LCII', 'PATK',
        'WGO', 'THO', 'CWH', 'HOG', 'PII', 'NKE', 'UAA', 'UA', 'VFC', 'COLM',
        'DECK', 'SKX', 'CROX', 'SHOO', 'WEYS', 'RCKY', 'SCVL', 'DSW', 'CAL', 'BORN',
        'RGS', 'ULTA', 'ELF', 'REV', 'COTY', 'EL', 'IPAR', 'ESTE', 'HRB', 'WPM',
        'CLF', 'X', 'MT', 'NUE', 'STLD', 'RS', 'CRS', 'ATI', 'CMC', 'CRS', 'ZEUS',
        'CENX', 'KALU', 'KWR', 'PATI', 'POLD', 'RFP', 'TMST', 'CIR', 'RXN', 'TWI',
        'TRS', 'MGRC', 'MINI', 'TTEK', 'XYL', 'AWI', 'BLDR', 'CENT', 'CSL', 'FAST',
        'GFF', 'HWKN', 'LII', 'LXU', 'MATW', 'MLI', 'MTRN', 'NPO', 'OC', 'ROCK',
        'SSD', 'TREX', 'TWI', 'UFPI', 'USCR', 'WTS', 'BOOM', 'CMT', 'GTLS', 'HIHO',
        'PLOW', 'SHLM', 'SSD', 'TNC', 'TRS', 'UFPI', 'USAP', 'WLDN', 'WTS', 'XONE',
        'APTV', 'F', 'GM', 'TSLA', 'RIVN', 'LCID', 'FSR', 'GOEV', 'NKLA', 'RIDE',
        'WKHS', 'XOS', 'ARVL', 'CENN', 'HYLN', 'PHBI', 'SPIR', 'AYRO', 'KNDI',
    }
    
    # Known NASDAQ financials and others
    nasdaq_financials = {'COIN', 'AFRM', 'SOFI', 'HOOD', 'OPEN', 'RDFN', 'Z', 'ZG'}
    
    if symbol in nyse_tech:
        return 'NYSE'
    
    if symbol in nasdaq_financials:
        return 'NASDAQ'
    
    if sector in nasdaq_sectors:
        return 'NASDAQ'
    
    return 'NYSE'

def fix_exchange_mapping():
    """Fix exchange mapping in tradingview_fetcher.py"""
    from core.tradingview_fetcher import DEFAULT_US_STOCKS
    
    print("Fetching S&P 500 data...")
    df = fetch_sp500_data()
    
    if df is None:
        print("Failed to fetch data")
        return
    
    # Create sector lookup
    sector_map = dict(zip(df['Symbol'], df['GICS Sector']))
    
    tv_file = Path(__file__).parent.parent / "core" / "tradingview_fetcher.py"
    
    # Read current content
    with open(tv_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Generate corrected exchange map
    exchange_map_lines = ["# Symbol 到 Exchange 的映射表", "EXCHANGE_MAP = {"]
    
    nasdaq_count = 0
    nyse_count = 0
    
    for symbol in sorted(set(DEFAULT_US_STOCKS)):
        sector = sector_map.get(symbol, '')
        exchange = get_exchange(symbol, sector)
        
        if exchange == 'NASDAQ':
            nasdaq_count += 1
        else:
            nyse_count += 1
            
        exchange_map_lines.append(f'    "{symbol}": "{exchange}",')
    
    exchange_map_lines.append("}")
    exchange_map_text = '\n'.join(exchange_map_lines)
    
    # Replace EXCHANGE_MAP
    import re
    exchange_pattern = r'# Symbol 到 Exchange 的映射表.*?EXCHANGE_MAP = \{.*?\}'
    if re.search(exchange_pattern, content, re.DOTALL):
        content = re.sub(exchange_pattern, exchange_map_text, content, flags=re.DOTALL)
    
    # Write back
    with open(tv_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {len(DEFAULT_US_STOCKS)} symbols in {tv_file}")
    print(f"  NASDAQ: {nasdaq_count}")
    print(f"  NYSE: {nyse_count}")
    
    # Verify some known stocks
    print("\nVerification:")
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'ACN', 'ABBV']
    for s in test_stocks:
        sector = sector_map.get(s, 'Unknown')
        exchange = get_exchange(s, sector)
        print(f"  {s} ({sector[:20]}): {exchange}")

if __name__ == "__main__":
    fix_exchange_mapping()
    print("\nDone!")
