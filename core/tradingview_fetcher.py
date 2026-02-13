"""
TradingView Data Fetcher Module

提供从 TradingView 获取美股历史数据的统一接口。
支持无账号/有账号两种模式，有账号可以获取更多历史数据。
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
import time
import asyncio
from functools import lru_cache

# Import tvdatafeed (optional — only needed for TradingViewFetcher)
try:
    from tvDatafeed import TvDatafeed, Interval
    HAS_TVDATAFEED = True
except ImportError:
    HAS_TVDATAFEED = False
    TvDatafeed = None
    Interval = None


# Symbol 到 Exchange 的映射表
EXCHANGE_MAP = {
    "A": "NYSE",
    "AAPL": "NASDAQ",
    "ABBV": "NYSE",
    "ABNB": "NYSE",
    "ABT": "NYSE",
    "ACGL": "NYSE",
    "ACN": "NYSE",
    "ADBE": "NASDAQ",
    "ADI": "NASDAQ",
    "ADM": "NYSE",
    "ADP": "NYSE",
    "ADSK": "NASDAQ",
    "AEE": "NYSE",
    "AEP": "NYSE",
    "AES": "NYSE",
    "AFL": "NYSE",
    "AIG": "NYSE",
    "AIZ": "NYSE",
    "AJG": "NYSE",
    "AKAM": "NASDAQ",
    "ALB": "NYSE",
    "ALGN": "NYSE",
    "ALL": "NYSE",
    "ALLE": "NYSE",
    "AMAT": "NASDAQ",
    "AMCR": "NYSE",
    "AMD": "NASDAQ",
    "AME": "NYSE",
    "AMGN": "NYSE",
    "AMP": "NYSE",
    "AMT": "NYSE",
    "AMZN": "NYSE",
    "ANET": "NASDAQ",
    "AON": "NYSE",
    "AOS": "NYSE",
    "APA": "NYSE",
    "APD": "NYSE",
    "APH": "NASDAQ",
    "APO": "NYSE",
    "APTV": "NYSE",
    "ARE": "NYSE",
    "ATO": "NYSE",
    "AVB": "NYSE",
    "AVGO": "NASDAQ",
    "AVY": "NYSE",
    "AWK": "NYSE",
    "AXON": "NYSE",
    "AXP": "NYSE",
    "AZO": "NYSE",
    "BA": "NYSE",
    "BAC": "NYSE",
    "BALL": "NYSE",
    "BAX": "NYSE",
    "BBY": "NYSE",
    "BDX": "NYSE",
    "BEN": "NYSE",
    "BF.B": "NYSE",
    "BG": "NYSE",
    "BIIB": "NYSE",
    "BK": "NYSE",
    "BKNG": "NYSE",
    "BKR": "NYSE",
    "BLDR": "NYSE",
    "BLK": "NYSE",
    "BMY": "NYSE",
    "BR": "NYSE",
    "BRK.B": "NYSE",
    "BRO": "NYSE",
    "BSX": "NYSE",
    "BX": "NYSE",
    "BXP": "NYSE",
    "C": "NYSE",
    "CAG": "NYSE",
    "CAH": "NYSE",
    "CARR": "NYSE",
    "CAT": "NYSE",
    "CB": "NYSE",
    "CBOE": "NYSE",
    "CBRE": "NYSE",
    "CCI": "NYSE",
    "CCL": "NYSE",
    "CDNS": "NASDAQ",
    "CDW": "NASDAQ",
    "CEG": "NYSE",
    "CF": "NYSE",
    "CFG": "NYSE",
    "CHD": "NYSE",
    "CHRW": "NYSE",
    "CHTR": "NASDAQ",
    "CI": "NYSE",
    "CINF": "NYSE",
    "CL": "NYSE",
    "CLX": "NYSE",
    "CMCSA": "NASDAQ",
    "CME": "NYSE",
    "CMG": "NYSE",
    "CMI": "NYSE",
    "CMS": "NYSE",
    "CNC": "NYSE",
    "CNP": "NYSE",
    "COF": "NYSE",
    "COIN": "NASDAQ",
    "COO": "NYSE",
    "COP": "NYSE",
    "COR": "NYSE",
    "COST": "NYSE",
    "CPAY": "NYSE",
    "CPB": "NYSE",
    "CPRT": "NYSE",
    "CPT": "NYSE",
    "CRL": "NYSE",
    "CRM": "NYSE",
    "CRWD": "NASDAQ",
    "CSCO": "NASDAQ",
    "CSGP": "NYSE",
    "CSX": "NYSE",
    "CTAS": "NYSE",
    "CTRA": "NYSE",
    "CTSH": "NYSE",
    "CTVA": "NYSE",
    "CVS": "NYSE",
    "CVX": "NYSE",
    "CZR": "NYSE",
    "D": "NYSE",
    "DAL": "NYSE",
    "DASH": "NYSE",
    "DAY": "NYSE",
    "DD": "NYSE",
    "DDOG": "NASDAQ",
    "DE": "NYSE",
    "DECK": "NYSE",
    "DELL": "NASDAQ",
    "DG": "NYSE",
    "DGX": "NYSE",
    "DHI": "NYSE",
    "DHR": "NYSE",
    "DIS": "NASDAQ",
    "DLR": "NYSE",
    "DLTR": "NYSE",
    "DOC": "NYSE",
    "DOV": "NYSE",
    "DOW": "NYSE",
    "DPZ": "NYSE",
    "DRI": "NYSE",
    "DTE": "NYSE",
    "DUK": "NYSE",
    "DVA": "NYSE",
    "DVN": "NYSE",
    "DXCM": "NYSE",
    "EA": "NASDAQ",
    "EBAY": "NYSE",
    "ECL": "NYSE",
    "ED": "NYSE",
    "EFX": "NYSE",
    "EG": "NYSE",
    "EIX": "NYSE",
    "EL": "NYSE",
    "ELV": "NYSE",
    "EMN": "NYSE",
    "EMR": "NYSE",
    "ENPH": "NASDAQ",
    "EOG": "NYSE",
    "EPAM": "NYSE",
    "EQIX": "NYSE",
    "EQR": "NYSE",
    "EQT": "NYSE",
    "ERIE": "NYSE",
    "ES": "NYSE",
    "ESS": "NYSE",
    "ETN": "NYSE",
    "ETR": "NYSE",
    "EVRG": "NYSE",
    "EW": "NYSE",
    "EXC": "NYSE",
    "EXE": "NYSE",
    "EXPD": "NYSE",
    "EXPE": "NYSE",
    "EXR": "NYSE",
    "F": "NYSE",
    "FANG": "NYSE",
    "FAST": "NYSE",
    "FCX": "NYSE",
    "FDS": "NYSE",
    "FDX": "NYSE",
    "FE": "NYSE",
    "FFIV": "NASDAQ",
    "FI": "NYSE",
    "FICO": "NASDAQ",
    "FIS": "NYSE",
    "FITB": "NYSE",
    "FOX": "NASDAQ",
    "FOXA": "NASDAQ",
    "FRT": "NYSE",
    "FSLR": "NASDAQ",
    "FTNT": "NASDAQ",
    "FTV": "NYSE",
    "GD": "NYSE",
    "GDDY": "NASDAQ",
    "GE": "NYSE",
    "GEHC": "NYSE",
    "GEN": "NASDAQ",
    "GEV": "NYSE",
    "GILD": "NYSE",
    "GIS": "NYSE",
    "GL": "NYSE",
    "GLW": "NASDAQ",
    "GM": "NYSE",
    "GNRC": "NYSE",
    "GOOG": "NASDAQ",
    "GOOGL": "NASDAQ",
    "GPC": "NYSE",
    "GPN": "NYSE",
    "GRMN": "NYSE",
    "GS": "NYSE",
    "GWW": "NYSE",
    "HAL": "NYSE",
    "HAS": "NYSE",
    "HBAN": "NYSE",
    "HCA": "NYSE",
    "HD": "NYSE",
    "HIG": "NYSE",
    "HII": "NYSE",
    "HLT": "NYSE",
    "HOLX": "NYSE",
    "HON": "NYSE",
    "HPE": "NASDAQ",
    "HPQ": "NASDAQ",
    "HRL": "NYSE",
    "HSIC": "NYSE",
    "HST": "NYSE",
    "HSY": "NYSE",
    "HUBB": "NYSE",
    "HUM": "NYSE",
    "HWM": "NYSE",
    "IBM": "NYSE",
    "ICE": "NYSE",
    "IDXX": "NYSE",
    "IEX": "NYSE",
    "IFF": "NYSE",
    "INCY": "NYSE",
    "INTC": "NASDAQ",
    "INTU": "NYSE",
    "INVH": "NYSE",
    "IP": "NYSE",
    "IPG": "NASDAQ",
    "IQV": "NYSE",
    "IR": "NYSE",
    "IRM": "NYSE",
    "ISRG": "NYSE",
    "IT": "NYSE",
    "ITW": "NYSE",
    "IVZ": "NYSE",
    "J": "NYSE",
    "JBHT": "NYSE",
    "JBL": "NASDAQ",
    "JCI": "NYSE",
    "JKHY": "NYSE",
    "JNJ": "NYSE",
    "JPM": "NYSE",
    "K": "NYSE",
    "KDP": "NYSE",
    "KEY": "NYSE",
    "KEYS": "NASDAQ",
    "KHC": "NYSE",
    "KIM": "NYSE",
    "KKR": "NYSE",
    "KLAC": "NASDAQ",
    "KMB": "NYSE",
    "KMI": "NYSE",
    "KMX": "NYSE",
    "KO": "NYSE",
    "KR": "NYSE",
    "KVUE": "NYSE",
    "L": "NYSE",
    "LDOS": "NYSE",
    "LEN": "NYSE",
    "LH": "NYSE",
    "LHX": "NYSE",
    "LII": "NYSE",
    "LIN": "NYSE",
    "LKQ": "NYSE",
    "LLY": "NYSE",
    "LMT": "NYSE",
    "LNT": "NYSE",
    "LOW": "NYSE",
    "LRCX": "NASDAQ",
    "LULU": "NYSE",
    "LUV": "NYSE",
    "LVS": "NYSE",
    "LW": "NYSE",
    "LYB": "NYSE",
    "LYV": "NASDAQ",
    "MA": "NYSE",
    "MAA": "NYSE",
    "MAR": "NYSE",
    "MAS": "NYSE",
    "MCD": "NYSE",
    "MCHP": "NASDAQ",
    "MCK": "NYSE",
    "MCO": "NYSE",
    "MDLZ": "NYSE",
    "MDT": "NYSE",
    "MET": "NYSE",
    "META": "NASDAQ",
    "MGM": "NYSE",
    "MHK": "NYSE",
    "MKC": "NYSE",
    "MKTX": "NYSE",
    "MLM": "NYSE",
    "MMC": "NYSE",
    "MMM": "NYSE",
    "MNST": "NYSE",
    "MO": "NYSE",
    "MOH": "NYSE",
    "MOS": "NYSE",
    "MPC": "NYSE",
    "MPWR": "NASDAQ",
    "MRK": "NYSE",
    "MRNA": "NYSE",
    "MS": "NYSE",
    "MSCI": "NYSE",
    "MSFT": "NASDAQ",
    "MSI": "NASDAQ",
    "MTB": "NYSE",
    "MTCH": "NASDAQ",
    "MTD": "NYSE",
    "MU": "NASDAQ",
    "NCLH": "NYSE",
    "NDAQ": "NYSE",
    "NDSN": "NYSE",
    "NEE": "NYSE",
    "NEM": "NYSE",
    "NFLX": "NASDAQ",
    "NI": "NYSE",
    "NKE": "NYSE",
    "NOC": "NYSE",
    "NOW": "NASDAQ",
    "NRG": "NYSE",
    "NSC": "NYSE",
    "NTAP": "NASDAQ",
    "NTRS": "NYSE",
    "NUE": "NYSE",
    "NVDA": "NASDAQ",
    "NVR": "NYSE",
    "NWS": "NASDAQ",
    "NWSA": "NASDAQ",
    "NXPI": "NASDAQ",
    "O": "NYSE",
    "ODFL": "NYSE",
    "OKE": "NYSE",
    "OMC": "NASDAQ",
    "ON": "NASDAQ",
    "ORCL": "NYSE",
    "ORLY": "NYSE",
    "OTIS": "NYSE",
    "OXY": "NYSE",
    "PANW": "NASDAQ",
    "PAYC": "NYSE",
    "PAYX": "NYSE",
    "PCAR": "NYSE",
    "PCG": "NYSE",
    "PEG": "NYSE",
    "PEP": "NYSE",
    "PFE": "NYSE",
    "PFG": "NYSE",
    "PG": "NYSE",
    "PGR": "NYSE",
    "PH": "NYSE",
    "PHM": "NYSE",
    "PKG": "NYSE",
    "PLD": "NYSE",
    "PLTR": "NASDAQ",
    "PM": "NYSE",
    "PNC": "NYSE",
    "PNR": "NYSE",
    "PNW": "NYSE",
    "PODD": "NYSE",
    "POOL": "NYSE",
    "PPG": "NYSE",
    "PPL": "NYSE",
    "PRU": "NYSE",
    "PSA": "NYSE",
    "PSKY": "NASDAQ",
    "PSX": "NYSE",
    "PTC": "NASDAQ",
    "PWR": "NYSE",
    "PYPL": "NYSE",
    "QCOM": "NASDAQ",
    "RCL": "NYSE",
    "REG": "NYSE",
    "REGN": "NYSE",
    "RF": "NYSE",
    "RJF": "NYSE",
    "RL": "NYSE",
    "RMD": "NYSE",
    "ROK": "NYSE",
    "ROL": "NYSE",
    "ROP": "NYSE",
    "ROST": "NYSE",
    "RSG": "NYSE",
    "RTX": "NYSE",
    "RVTY": "NYSE",
    "SBAC": "NYSE",
    "SBUX": "NYSE",
    "SCHW": "NYSE",
    "SHW": "NYSE",
    "SJM": "NYSE",
    "SLB": "NYSE",
    "SMCI": "NASDAQ",
    "SNA": "NYSE",
    "SNPS": "NASDAQ",
    "SO": "NYSE",
    "SOLV": "NYSE",
    "SPG": "NYSE",
    "SPGI": "NYSE",
    "SRE": "NYSE",
    "STE": "NYSE",
    "STLD": "NYSE",
    "STT": "NYSE",
    "STX": "NASDAQ",
    "STZ": "NYSE",
    "SW": "NYSE",
    "SWK": "NYSE",
    "SWKS": "NASDAQ",
    "SYF": "NYSE",
    "SYK": "NYSE",
    "SYY": "NYSE",
    "T": "NASDAQ",
    "TAP": "NYSE",
    "TDG": "NYSE",
    "TDY": "NASDAQ",
    "TECH": "NYSE",
    "TEL": "NASDAQ",
    "TER": "NASDAQ",
    "TFC": "NYSE",
    "TGT": "NYSE",
    "TJX": "NYSE",
    "TKO": "NASDAQ",
    "TMO": "NYSE",
    "TMUS": "NASDAQ",
    "TPL": "NYSE",
    "TPR": "NYSE",
    "TRGP": "NYSE",
    "TRMB": "NASDAQ",
    "TROW": "NYSE",
    "TRV": "NYSE",
    "TSCO": "NYSE",
    "TSLA": "NYSE",
    "TSN": "NYSE",
    "TT": "NYSE",
    "TTD": "NASDAQ",
    "TTWO": "NASDAQ",
    "TXN": "NASDAQ",
    "TXT": "NYSE",
    "TYL": "NASDAQ",
    "UAL": "NYSE",
    "UBER": "NYSE",
    "UDR": "NYSE",
    "UHS": "NYSE",
    "ULTA": "NYSE",
    "UNH": "NYSE",
    "UNP": "NYSE",
    "UPS": "NYSE",
    "URI": "NYSE",
    "USB": "NYSE",
    "V": "NYSE",
    "VICI": "NYSE",
    "VLO": "NYSE",
    "VLTO": "NYSE",
    "VMC": "NYSE",
    "VRSK": "NYSE",
    "VRSN": "NASDAQ",
    "VRTX": "NYSE",
    "VST": "NYSE",
    "VTR": "NYSE",
    "VTRS": "NYSE",
    "VZ": "NASDAQ",
    "WAB": "NYSE",
    "WAT": "NYSE",
    "WBA": "NYSE",
    "WBD": "NASDAQ",
    "WDAY": "NASDAQ",
    "WDC": "NASDAQ",
    "WEC": "NYSE",
    "WELL": "NYSE",
    "WFC": "NYSE",
    "WM": "NYSE",
    "WMB": "NYSE",
    "WMT": "NYSE",
    "WRB": "NYSE",
    "WSM": "NYSE",
    "WST": "NYSE",
    "WTW": "NYSE",
    "WY": "NYSE",
    "WYNN": "NYSE",
    "XEL": "NYSE",
    "XOM": "NYSE",
    "XYL": "NYSE",
    "XYZ": "NYSE",
    "YUM": "NYSE",
    "ZBH": "NYSE",
    "ZBRA": "NASDAQ",
    "ZTS": "NYSE",
}

# S&P 500 成分股列表（共503只）
DEFAULT_US_STOCKS = [

    # Row 1
    "MMM",
    "AOS",
    "ABT",
    "ABBV",
    "ACN",
    "ADBE",
    "AMD",
    "AES",
    "AFL",
    "A",

    # Row 2
    "APD",
    "ABNB",
    "AKAM",
    "ALB",
    "ARE",
    "ALGN",
    "ALLE",
    "LNT",
    "ALL",
    "GOOGL",

    # Row 3
    "GOOG",
    "MO",
    "AMZN",
    "AMCR",
    "AEE",
    "AEP",
    "AXP",
    "AIG",
    "AMT",
    "AWK",

    # Row 4
    "AMP",
    "AME",
    "AMGN",
    "APH",
    "ADI",
    "AON",
    "APA",
    "APO",
    "AAPL",
    "AMAT",

    # Row 5
    "APTV",
    "ACGL",
    "ADM",
    "ANET",
    "AJG",
    "AIZ",
    "T",
    "ATO",
    "ADSK",
    "ADP",

    # Row 6
    "AZO",
    "AVB",
    "AVY",
    "AXON",
    "BKR",
    "BALL",
    "BAC",
    "BAX",
    "BDX",
    "BRK.B",

    # Row 7
    "BBY",
    "TECH",
    "BIIB",
    "BLK",
    "BX",
    "XYZ",
    "BK",
    "BA",
    "BKNG",
    "BSX",

    # Row 8
    "BMY",
    "AVGO",
    "BR",
    "BRO",
    "BF.B",
    "BLDR",
    "BG",
    "BXP",
    "CHRW",
    "CDNS",

    # Row 9
    "CZR",
    "CPT",
    "CPB",
    "COF",
    "CAH",
    "KMX",
    "CCL",
    "CARR",
    "CAT",
    "CBOE",

    # Row 10
    "CBRE",
    "CDW",
    "COR",
    "CNC",
    "CNP",
    "CF",
    "CRL",
    "SCHW",
    "CHTR",
    "CVX",

    # Row 11
    "CMG",
    "CB",
    "CHD",
    "CI",
    "CINF",
    "CTAS",
    "CSCO",
    "C",
    "CFG",
    "CLX",

    # Row 12
    "CME",
    "CMS",
    "KO",
    "CTSH",
    "COIN",
    "CL",
    "CMCSA",
    "CAG",
    "COP",
    "ED",

    # Row 13
    "STZ",
    "CEG",
    "COO",
    "CPRT",
    "GLW",
    "CPAY",
    "CTVA",
    "CSGP",
    "COST",
    "CTRA",

    # Row 14
    "CRWD",
    "CCI",
    "CSX",
    "CMI",
    "CVS",
    "DHR",
    "DRI",
    "DDOG",
    "DVA",
    "DAY",

    # Row 15
    "DECK",
    "DE",
    "DELL",
    "DAL",
    "DVN",
    "DXCM",
    "FANG",
    "DLR",
    "DG",
    "DLTR",

    # Row 16
    "D",
    "DPZ",
    "DASH",
    "DOV",
    "DOW",
    "DHI",
    "DTE",
    "DUK",
    "DD",
    "EMN",

    # Row 17
    "ETN",
    "EBAY",
    "ECL",
    "EIX",
    "EW",
    "EA",
    "ELV",
    "EMR",
    "ENPH",
    "ETR",

    # Row 18
    "EOG",
    "EPAM",
    "EQT",
    "EFX",
    "EQIX",
    "EQR",
    "ERIE",
    "ESS",
    "EL",
    "EG",

    # Row 19
    "EVRG",
    "ES",
    "EXC",
    "EXE",
    "EXPE",
    "EXPD",
    "EXR",
    "XOM",
    "FFIV",
    "FDS",

    # Row 20
    "FICO",
    "FAST",
    "FRT",
    "FDX",
    "FIS",
    "FITB",
    "FSLR",
    "FE",
    "FI",
    "F",

    # Row 21
    "FTNT",
    "FTV",
    "FOXA",
    "FOX",
    "BEN",
    "FCX",
    "GRMN",
    "IT",
    "GE",
    "GEHC",

    # Row 22
    "GEV",
    "GEN",
    "GNRC",
    "GD",
    "GIS",
    "GM",
    "GPC",
    "GILD",
    "GPN",
    "GL",

    # Row 23
    "GDDY",
    "GS",
    "HAL",
    "HIG",
    "HAS",
    "HCA",
    "DOC",
    "HSIC",
    "HSY",
    "HPE",

    # Row 24
    "HLT",
    "HOLX",
    "HD",
    "HON",
    "HRL",
    "HST",
    "HWM",
    "HPQ",
    "HUBB",
    "HUM",

    # Row 25
    "HBAN",
    "HII",
    "IBM",
    "IEX",
    "IDXX",
    "ITW",
    "INCY",
    "IR",
    "PODD",
    "INTC",

    # Row 26
    "ICE",
    "IFF",
    "IP",
    "IPG",
    "INTU",
    "ISRG",
    "IVZ",
    "INVH",
    "IQV",
    "IRM",

    # Row 27
    "JBHT",
    "JBL",
    "JKHY",
    "J",
    "JNJ",
    "JCI",
    "JPM",
    "K",
    "KVUE",
    "KDP",

    # Row 28
    "KEY",
    "KEYS",
    "KMB",
    "KIM",
    "KMI",
    "KKR",
    "KLAC",
    "KHC",
    "KR",
    "LHX",

    # Row 29
    "LH",
    "LRCX",
    "LW",
    "LVS",
    "LDOS",
    "LEN",
    "LII",
    "LLY",
    "LIN",
    "LYV",

    # Row 30
    "LKQ",
    "LMT",
    "L",
    "LOW",
    "LULU",
    "LYB",
    "MTB",
    "MPC",
    "MKTX",
    "MAR",

    # Row 31
    "MMC",
    "MLM",
    "MAS",
    "MA",
    "MTCH",
    "MKC",
    "MCD",
    "MCK",
    "MDT",
    "MRK",

    # Row 32
    "META",
    "MET",
    "MTD",
    "MGM",
    "MCHP",
    "MU",
    "MSFT",
    "MAA",
    "MRNA",
    "MHK",

    # Row 33
    "MOH",
    "TAP",
    "MDLZ",
    "MPWR",
    "MNST",
    "MCO",
    "MS",
    "MOS",
    "MSI",
    "MSCI",

    # Row 34
    "NDAQ",
    "NTAP",
    "NFLX",
    "NEM",
    "NWSA",
    "NWS",
    "NEE",
    "NKE",
    "NI",
    "NDSN",

    # Row 35
    "NSC",
    "NTRS",
    "NOC",
    "NCLH",
    "NRG",
    "NUE",
    "NVDA",
    "NVR",
    "NXPI",
    "ORLY",

    # Row 36
    "OXY",
    "ODFL",
    "OMC",
    "ON",
    "OKE",
    "ORCL",
    "OTIS",
    "PCAR",
    "PKG",
    "PLTR",

    # Row 37
    "PANW",
    "PSKY",
    "PH",
    "PAYX",
    "PAYC",
    "PYPL",
    "PNR",
    "PEP",
    "PFE",
    "PCG",

    # Row 38
    "PM",
    "PSX",
    "PNW",
    "PNC",
    "POOL",
    "PPG",
    "PPL",
    "PFG",
    "PG",
    "PGR",

    # Row 39
    "PLD",
    "PRU",
    "PEG",
    "PTC",
    "PSA",
    "PHM",
    "PWR",
    "QCOM",
    "DGX",
    "RL",

    # Row 40
    "RJF",
    "RTX",
    "O",
    "REG",
    "REGN",
    "RF",
    "RSG",
    "RMD",
    "RVTY",
    "ROK",

    # Row 41
    "ROL",
    "ROP",
    "ROST",
    "RCL",
    "SPGI",
    "CRM",
    "SBAC",
    "SLB",
    "STX",
    "SRE",

    # Row 42
    "NOW",
    "SHW",
    "SPG",
    "SWKS",
    "SJM",
    "SW",
    "SNA",
    "SOLV",
    "SO",
    "LUV",

    # Row 43
    "SWK",
    "SBUX",
    "STT",
    "STLD",
    "STE",
    "SYK",
    "SMCI",
    "SYF",
    "SNPS",
    "SYY",

    # Row 44
    "TMUS",
    "TROW",
    "TTWO",
    "TPR",
    "TRGP",
    "TGT",
    "TEL",
    "TDY",
    "TER",
    "TSLA",

    # Row 45
    "TXN",
    "TPL",
    "TXT",
    "TMO",
    "TJX",
    "TKO",
    "TTD",
    "TSCO",
    "TT",
    "TDG",

    # Row 46
    "TRV",
    "TRMB",
    "TFC",
    "TYL",
    "TSN",
    "USB",
    "UBER",
    "UDR",
    "ULTA",
    "UNP",

    # Row 47
    "UAL",
    "UPS",
    "URI",
    "UNH",
    "UHS",
    "VLO",
    "VTR",
    "VLTO",
    "VRSN",
    "VRSK",

    # Row 48
    "VZ",
    "VRTX",
    "VTRS",
    "VICI",
    "V",
    "VST",
    "VMC",
    "WRB",
    "GWW",
    "WAB",

    # Row 49
    "WBA",
    "WMT",
    "DIS",
    "WBD",
    "WM",
    "WAT",
    "WEC",
    "WFC",
    "WELL",
    "WST",

    # Row 50
    "WDC",
    "WY",
    "WSM",
    "WMB",
    "WTW",
    "WDAY",
    "WYNN",
    "XEL",
    "XYL",
    "YUM",

    # Row 51
    "ZBRA",
    "ZBH",
    "ZTS",
]


class TradingViewFetcher:
    """
    TradingView 数据获取器

    提供同步和异步接口获取美股历史数据。
    支持 WebSocket 自动重连。
    """

    INTERVAL_MAP = (
        {
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
        if HAS_TVDATAFEED
        else {}
    )

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl: int = 3600,
    ):
        """
        初始化 TradingView 数据获取器

        Args:
            username: TradingView 用户名（可选，可获取更多数据）
            password: TradingView 密码（可选）
            cache_dir: 缓存目录（可选）
            cache_ttl: 缓存时间（秒）
        """
        if not HAS_TVDATAFEED:
            raise ImportError("tvDatafeed not installed. Run: pip install tradingview-datafeed")
        self._username = username
        self._password = password
        self.tv = TvDatafeed(username, password)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        self._memory_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._consecutive_failures = 0

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _reconnect(self):
        """Recreate the TvDatafeed WebSocket connection."""
        try:
            self.tv = TvDatafeed(self._username, self._password)
            self._consecutive_failures = 0
        except Exception:
            pass
    
    def get_exchange(self, symbol: str) -> str:
        """获取 symbol 对应的交易所"""
        return EXCHANGE_MAP.get(symbol.upper(), "NASDAQ")
    
    def _get_cache_key(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        n_bars: int,
    ) -> str:
        """生成缓存 key"""
        return f"{exchange}_{symbol}_{interval}_{n_bars}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从内存或磁盘缓存获取数据"""
        # 内存缓存
        if cache_key in self._memory_cache:
            data, timestamp = self._memory_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data.copy()
            else:
                del self._memory_cache[cache_key]
        
        # 磁盘缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            if cache_file.exists():
                mtime = cache_file.stat().st_mtime
                if time.time() - mtime < self.cache_ttl:
                    return pd.read_parquet(cache_file)
        
        return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame):
        """保存到缓存"""
        # 内存缓存
        self._memory_cache[cache_key] = (df.copy(), time.time())
        
        # 磁盘缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            df.to_parquet(cache_file)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        interval: str = "1d",
        n_bars: int = 10000,
        use_cache: bool = True,
        retry: int = 3,
        delay: float = 1.0,
    ) -> Optional[pd.DataFrame]:
        """
        获取 OHLCV 数据
        
        Args:
            symbol: 股票代码
            exchange: 交易所代码（None 则自动检测）
            interval: 时间周期 (1m, 5m, 1h, 1d, 1W, 1M)
            n_bars: 获取的 K 线数量（最大 50000，登录后可获取更多）
            use_cache: 是否使用缓存
            retry: 重试次数
            delay: 重试延迟（秒）
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        symbol = symbol.upper()
        if exchange is None:
            exchange = self.get_exchange(symbol)
        
        cache_key = self._get_cache_key(symbol, exchange, interval, n_bars)
        
        # 检查缓存
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        tv_interval = self.INTERVAL_MAP.get(interval, Interval.in_daily)
        
        for attempt in range(retry):
            try:
                # Reconnect if we've had consecutive failures
                if self._consecutive_failures >= 2:
                    self._reconnect()
                    time.sleep(2)

                df = self.tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=tv_interval,
                    n_bars=min(n_bars, 50000),
                )

                if df is None or df.empty:
                    self._consecutive_failures += 1
                    if attempt < retry - 1:
                        # Reconnect on empty result (likely connection issue)
                        if self._consecutive_failures >= 2:
                            self._reconnect()
                        time.sleep(delay * (attempt + 1))
                        continue
                    return None

                # Success — reset failure counter
                self._consecutive_failures = 0

                # 标准化列名
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })

                # 确保索引是 datetime
                df.index = pd.to_datetime(df.index)
                df.index.name = 'Date'

                # 保存到缓存
                if use_cache:
                    self._save_to_cache(cache_key, df)

                return df

            except Exception as e:
                self._consecutive_failures += 1
                error_msg = str(e).lower()
                # Reconnect on connection errors
                if any(kw in error_msg for kw in ['timeout', 'connection', 'lost', 'closed', 'websocket']):
                    self._reconnect()

                if attempt < retry - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    print(f"Error fetching {symbol}: {e}")
                    return None

        return None
    
    def fetch_history(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        exchange: Optional[str] = None,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        获取指定日期范围的历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期（默认今天）
            exchange: 交易所代码
            interval: 时间周期
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 计算需要的 bar 数量
        if interval == "1d":
            days = (end_date - start_date).days
            n_bars = min(days + 500, 50000)  # 加缓冲
        elif interval == "1W":
            weeks = (end_date - start_date).days // 7
            n_bars = min(weeks + 100, 50000)
        elif interval == "1M":
            months = (end_date - start_date).days // 30
            n_bars = min(months + 24, 50000)
        else:
            n_bars = 50000
        
        # 获取数据
        df = self.fetch_ohlcv(symbol, exchange, interval, n_bars)
        
        if df is None or df.empty:
            return None
        
        # 过滤日期范围
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        df = df[(df.index >= start_str) & (df.index <= end_str)]
        
        return df
    
    async def fetch_ohlcv_async(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        interval: str = "1d",
        n_bars: int = 10000,
    ) -> Optional[pd.DataFrame]:
        """异步获取 OHLCV 数据"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.fetch_ohlcv, symbol, exchange, interval, n_bars
        )
    
    async def fetch_batch(
        self,
        symbols: List[str],
        interval: str = "1d",
        n_bars: int = 10000,
        max_concurrent: int = 5,
        delay: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据
        
        Args:
            symbols: 股票代码列表
            interval: 时间周期
            n_bars: K线数量
            max_concurrent: 最大并发数
            delay: 请求间隔（秒）
            
        Returns:
            Dict[symbol, DataFrame]
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_one(symbol):
            async with semaphore:
                df = await self.fetch_ohlcv_async(symbol, interval=interval, n_bars=n_bars)
                if df is not None:
                    results[symbol] = df
                await asyncio.sleep(delay)
        
        tasks = [fetch_one(s) for s in symbols]
        await asyncio.gather(*tasks)
        
        return results


class USStockDatabase:
    """
    美股本地数据库管理器
    
    管理本地存储的美股历史数据，支持增量更新。
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        username: Optional[str] = None,
        password: Optional[str] = None,
        fetcher=None,
    ):
        """
        初始化数据库

        Args:
            data_dir: 数据存储目录
            username: TradingView 用户名（可选）
            password: TradingView 密码（可选）
            fetcher: 数据获取器实例（可选，需有 fetch_history 方法）
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if fetcher is not None:
            self.fetcher = fetcher
        elif HAS_TVDATAFEED:
            self.fetcher = TradingViewFetcher(username, password)
        else:
            self.fetcher = None  # Storage-only mode (no download)

        # 元数据文件
        self.meta_file = self.data_dir / "_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        import json
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存元数据"""
        import json
        with open(self.meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_stock_path(self, symbol: str) -> Path:
        """获取股票数据文件路径"""
        return self.data_dir / f"{symbol.upper()}.parquet"
    
    def list_stocks(self) -> List[str]:
        """列出本地已存储的所有股票"""
        return [f.stem for f in self.data_dir.glob("*.parquet") if not f.stem.startswith('_')]
    
    def load_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载本地股票数据"""
        path = self.get_stock_path(symbol)
        if path.exists():
            return pd.read_parquet(path)
        return None
    
    def save_stock(self, symbol: str, df: pd.DataFrame):
        """保存股票数据"""
        path = self.get_stock_path(symbol)
        
        # Ensure index is timezone-naive for consistent storage
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        
        df.to_parquet(path)
        
        # 更新元数据
        self.metadata[symbol.upper()] = {
            'last_update': datetime.now().isoformat(),
            'rows': len(df),
            'start_date': df.index[0].strftime("%Y-%m-%d"),
            'end_date': df.index[-1].strftime("%Y-%m-%d"),
        }
        self._save_metadata()
    
    def update_stock(
        self,
        symbol: str,
        start_date: str = "1990-01-01",
        force: bool = False,
    ) -> bool:
        """
        更新单只股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期（新数据）
            force: 强制重新下载
            
        Returns:
            是否成功
        """
        symbol = symbol.upper()
        
        # 检查本地数据
        if not force:
            existing = self.load_stock(symbol)
            if existing is not None and len(existing) > 0:
                last_date = existing.index[-1]
                if last_date >= pd.Timestamp.now() - pd.Timedelta(days=2):
                    print(f"  {symbol}: Up to date ({len(existing)} rows)")
                    return True
                # 增量更新
                start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        # 获取新数据
        print(f"  {symbol}: Fetching from {start_date}...", end=" ")
        df = self.fetcher.fetch_history(symbol, start_date)
        
        if df is None or df.empty:
            print("FAILED")
            return False
        
        # 合并数据
        if not force:
            existing = self.load_stock(symbol)
            if existing is not None:
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
        
        self.save_stock(symbol, df)
        print(f"OK ({len(df)} rows)")
        return True
    
    def update_all(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = "1990-01-01",
        force: bool = False,
        delay: float = 0.5,
    ):
        """
        更新所有股票数据
        
        Args:
            symbols: 股票列表（默认 DEFAULT_US_STOCKS）
            start_date: 开始日期
            force: 强制重新下载
            delay: 请求间隔
        """
        if symbols is None:
            symbols = DEFAULT_US_STOCKS
        
        print(f"=" * 60)
        print(f"Updating US Stock Database")
        print(f"=" * 60)
        print(f"Total symbols: {len(symbols)}")
        print(f"Data directory: {self.data_dir}")
        print(f"=" * 60)
        
        success = 0
        failed = []
        
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}]", end=" ")
            if self.update_stock(symbol, start_date, force):
                success += 1
            else:
                failed.append(symbol)
            
            if delay > 0:
                time.sleep(delay)
        
        print(f"=" * 60)
        print(f"Success: {success}/{len(symbols)}")
        if failed:
            print(f"Failed: {failed}")
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """获取股票元信息"""
        return self.metadata.get(symbol.upper())
    
    def get_database_summary(self) -> pd.DataFrame:
        """获取数据库摘要"""
        data = []
        for symbol, meta in self.metadata.items():
            if isinstance(meta, dict):
                data.append({
                    'symbol': symbol,
                    **meta
                })
        return pd.DataFrame(data)


# 便捷函数
def get_fetcher(username: Optional[str] = None, password: Optional[str] = None) -> TradingViewFetcher:
    """获取 TradingView 数据获取器实例"""
    return TradingViewFetcher(username, password)


def get_database(
    data_dir: Optional[Union[str, Path]] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> USStockDatabase:
    """获取美股数据库实例"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw" / "us"
    return USStockDatabase(data_dir, username, password)


# 测试代码
if __name__ == "__main__":
    # 测试获取数据
    fetcher = TradingViewFetcher()
    
    # 测试单只股票
    print("Testing AAPL...")
    df = fetcher.fetch_ohlcv("AAPL", n_bars=5000)
    if df is not None:
        print(f"  Got {len(df)} rows")
        print(df.tail())
    
    # 测试数据库
    print("\nTesting Database...")
    db = get_database()
    db.update_stock("MSFT")
    print(f"  Local stocks: {db.list_stocks()}")
