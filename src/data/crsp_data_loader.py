"""
CRSP Data Loader for Large-Scale US Stock Market Dataset

支持从CRSP (Center for Research in Security Prices) 获取：
- NYSE, AMEX, NASDAQ 所有普通股
- 时间范围：1993-2019年
- 包含OHLCV价格数据和移动平均线

Author: Research Team
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockMetadata:
    """股票元数据"""
    ticker: str
    permno: str  # CRSP永久编号
    exchange: str  # NYSE/AMEX/NASDAQ
    industry_code: str
    start_date: str
    end_date: str
    market_cap: float  # 市值（百万美元）


class CRSPLargeScaleDataset:
    """
    CRSP大规模美股数据集
    
    数据集规模：
    - 股票数量：约8,000+只（1993-2019年间所有上市普通股）
    - 时间跨度：27年（1993-2019）
    - 数据点：约5000万+个日度观测
    - 交易所覆盖：NYSE, AMEX, NASDAQ
    
    Attributes:
        data_dir: 数据存储目录
        start_year: 起始年份
        end_year: 结束年份
        min_price: 最低价格过滤（剔除仙股）
        min_market_cap: 最小市值过滤（百万美元）
    """
    
    # 行业板块分类（根据SIC代码）
    SECTOR_MAPPING = {
        'Consumer': {'sic_range': [(2000, 3999)], 'desc': '消费品'},
        'Manufacturing': {'sic_range': [(4000, 4999)], 'desc': '制造业'},
        'Technology': {'sic_range': [(3570, 3579), (3660, 3699), (3810, 3899), (7370, 7379)], 'desc': '科技'},
        'Healthcare': {'sic_range': [(2800, 2899), (8000, 8099)], 'desc': '医疗保健'},
        'Financials': {'sic_range': [(6000, 6999)], 'desc': '金融服务'},
        'Energy': {'sic_range': [(1200, 1399), (2900, 2999)], 'desc': '能源'},
        'Utilities': {'sic_range': [(4900, 4999)], 'desc': '公用事业'},
        'Transportation': {'sic_range': [(4000, 4799)], 'desc': '交通运输'},
        'Retail': {'sic_range': [(5200, 5999)], 'desc': '零售贸易'},
        'Services': {'sic_range': [(7000, 8999)], 'desc': '服务业'},
    }
    
    def __init__(
        self,
        data_dir: str = "data/crsp_large_scale",
        start_year: int = 1993,
        end_year: int = 2019,
        min_price: float = 1.0,
        min_market_cap: float = 10.0,  # 百万美元
        require_full_history: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.start_year = start_year
        self.end_year = end_year
        self.min_price = min_price
        self.min_market_cap = min_market_cap
        self.require_full_history = require_full_history
        
        self.stock_metadata: Dict[str, StockMetadata] = {}
        self.sector_stocks: Dict[str, List[str]] = {sector: [] for sector in self.SECTOR_MAPPING.keys()}
        
        logger.info(f"CRSP Dataset initialized: {start_year}-{end_year}")
        logger.info(f"Data directory: {self.data_dir}")
        
    def get_sector_from_sic(self, sic_code: int) -> str:
        """根据SIC代码确定行业板块"""
        for sector, info in self.SECTOR_MAPPING.items():
            for start, end in info['sic_range']:
                if start <= sic_code <= end:
                    return sector
        return 'Others'
    
    def download_from_wrds(
        self,
        wrds_username: str,
        batch_size: int = 1000,
        use_cache: bool = True
    ) -> None:
        """
        从WRDS (Wharton Research Data Services) 下载CRSP数据
        
        Args:
            wrds_username: WRDS账号
            batch_size: 每批处理股票数量
            use_cache: 是否使用本地缓存
        """
        try:
            import wrds
        except ImportError:
            logger.error("WRDS library not installed. Install with: pip install wrds")
            raise
        
        logger.info("Connecting to WRDS...")
        db = wrds.Connection(wrds_username=wrds_username)
        
        # 获取股票列表
        logger.info("Fetching stock list...")
        stock_query = f"""
        SELECT DISTINCT permno, ticker, exchcd, shrcd, siccd, 
               namedt, nameenddt, 
               ABS(prc) * shrout / 1000 as market_cap_millions
        FROM crsp.msf 
        WHERE date >= '{self.start_year}-01-01'
          AND date <= '{self.end_year}-12-31'
          AND shrcd IN (10, 11)  -- 普通股
          AND exchcd IN (1, 2, 3)  -- NYSE, AMEX, NASDAQ
        """
        
        stock_list = db.raw_sql(stock_query)
        logger.info(f"Found {len(stock_list)} stock records")
        
        # 筛选有效股票
        valid_stocks = self._filter_valid_stocks(stock_list)
        logger.info(f"Valid stocks after filtering: {len(valid_stocks)}")
        
        # 按板块分组
        self._classify_by_sector(valid_stocks)
        
        # 下载价格数据
        self._download_price_data(db, valid_stocks, batch_size, use_cache)
        
        db.close()
        logger.info("Data download complete!")
        
    def _filter_valid_stocks(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """筛选符合条件的股票"""
        # 去除重复（保留最新记录）
        stock_list = stock_list.drop_duplicates(subset=['permno'], keep='last')
        
        # 市值过滤
        if self.min_market_cap > 0:
            stock_list = stock_list[stock_list['market_cap_millions'] >= self.min_market_cap]
        
        # 交易所映射
        exch_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
        stock_list['exchange'] = stock_list['exchcd'].map(exch_map)
        
        logger.info(f"Exchange distribution:")
        logger.info(stock_list['exchange'].value_counts())
        
        return stock_list
    
    def _classify_by_sector(self, stock_list: pd.DataFrame) -> None:
        """按行业分类股票"""
        for _, row in stock_list.iterrows():
            sector = self.get_sector_from_sic(int(row['siccd'])) if pd.notna(row['siccd']) else 'Others'
            self.sector_stocks[sector].append(row['ticker'])
            
            self.stock_metadata[row['ticker']] = StockMetadata(
                ticker=row['ticker'],
                permno=str(row['permno']),
                exchange=row['exchange'],
                industry_code=str(int(row['siccd'])) if pd.notna(row['siccd']) else '9999',
                start_date=str(row['namedt']),
                end_date=str(row['nameenddt']),
                market_cap=row['market_cap_millions']
            )
        
        # 报告各板块股票数量
        logger.info("Sector distribution:")
        for sector, tickers in self.sector_stocks.items():
            if len(tickers) > 0:
                logger.info(f"  {sector}: {len(tickers)} stocks")
    
    def _download_price_data(
        self,
        db,
        stock_list: pd.DataFrame,
        batch_size: int,
        use_cache: bool
    ) -> None:
        """下载价格数据"""
        permnos = stock_list['permno'].unique().tolist()
        total_batches = (len(permnos) + batch_size - 1) // batch_size
        
        for i in range(0, len(permnos), batch_size):
            batch_permnos = permnos[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Downloading batch {batch_num}/{total_batches} ({len(batch_permnos)} stocks)...")
            
            permno_str = ','.join([str(p) for p in batch_permnos])
            
            query = f"""
            SELECT permno, date, ticker, 
                   ABS(prc) as close,  -- 收盘价
                   ABS(openprc) as open,  -- 开盘价
                   askhi as high,  -- 最高价
                   bidlo as low,  -- 最低价
                   vol as volume,  -- 成交量
                   shrout as shares_outstanding  -- 流通股数
            FROM crsp.msf
            WHERE permno IN ({permno_str})
              AND date >= '{self.start_year}-01-01'
              AND date <= '{self.end_year}-12-31'
            ORDER BY permno, date
            """
            
            try:
                batch_data = db.raw_sql(query)
                self._process_and_save_batch(batch_data, use_cache)
            except Exception as e:
                logger.error(f"Error downloading batch {batch_num}: {e}")
                continue
    
    def _process_and_save_batch(self, batch_data: pd.DataFrame, use_cache: bool) -> None:
        """处理并保存批次数据"""
        if batch_data.empty:
            return
        
        # 计算移动平均线
        batch_data = batch_data.sort_values(['ticker', 'date'])
        batch_data['ma20'] = batch_data.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean()
        )
        
        # 计算收益率
        batch_data['return'] = batch_data.groupby('ticker')['close'].pct_change()
        
        # 按股票保存
        for ticker in batch_data['ticker'].unique():
            ticker_data = batch_data[batch_data['ticker'] == ticker].copy()
            
            if len(ticker_data) < 252:  # 至少需要一年数据
                continue
            
            # 价格过滤
            if (ticker_data['close'] < self.min_price).any():
                continue
            
            # 保存为CSV
            file_path = self.data_dir / f"{ticker}.csv"
            
            if use_cache and file_path.exists():
                # 合并现有数据
                existing = pd.read_csv(file_path, parse_dates=['date'])
                ticker_data = pd.concat([existing, ticker_data]).drop_duplicates('date')
                ticker_data = ticker_data.sort_values('date')
            
            ticker_data.to_csv(file_path, index=False)
    
    def load_stock_data(
        self,
        ticker: str,
        window_size: int = 20,
        add_features: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        加载单只股票数据
        
        Args:
            ticker: 股票代码
            window_size: 用于计算技术指标的窗口
            add_features: 是否添加衍生特征
        
        Returns:
            DataFrame with OHLCV + MA + derived features
        """
        file_path = self.data_dir / f"{ticker}.csv"
        
        if not file_path.exists():
            logger.warning(f"Data not found for {ticker}")
            return None
        
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        if add_features:
            df = self._add_technical_features(df, window_size)
        
        return df
    
    def _add_technical_features(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """添加技术指标特征"""
        # 基础价格特征
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        # 移动平均线
        for ma_window in [5, 10, 20, 60]:
            df[f'ma{ma_window}'] = df['close'].rolling(window=ma_window, min_periods=ma_window//2).mean()
            df[f'ma{ma_window}_ratio'] = df['close'] / df[f'ma{ma_window}']
        
        # 波动率
        df['volatility'] = df['return'].rolling(window=window, min_periods=window//2).std()
        
        # 成交量特征
        df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # 动量指标
        for momentum_window in [5, 10, 20]:
            df[f'momentum_{momentum_window}'] = df['close'].pct_change(momentum_window)
        
        return df
    
    def get_sector_dataset(
        self,
        sector: str,
        min_stocks: int = 50,
        max_stocks: Optional[int] = None
    ) -> List[str]:
        """
        获取指定板块的股票列表
        
        Args:
            sector: 板块名称
            min_stocks: 该板块最少股票数
            max_stocks: 该板块最多股票数
        
        Returns:
            股票代码列表
        """
        if sector not in self.sector_stocks:
            logger.warning(f"Unknown sector: {sector}")
            return []
        
        stocks = self.sector_stocks[sector]
        
        if len(stocks) < min_stocks:
            logger.warning(f"Sector {sector} has only {len(stocks)} stocks (< {min_stocks})")
            return stocks
        
        if max_stocks and len(stocks) > max_stocks:
            # 按市值选择最大的股票
            stocks_with_cap = [(s, self.stock_metadata[s].market_cap) for s in stocks]
            stocks_with_cap.sort(key=lambda x: x[1], reverse=True)
            stocks = [s for s, _ in stocks_with_cap[:max_stocks]]
        
        return stocks
    
    def get_all_sectors(self) -> Dict[str, int]:
        """获取所有板块及其股票数量"""
        return {sector: len(tickers) for sector, tickers in self.sector_stocks.items()}
    
    def create_dataset_statistics(self) -> pd.DataFrame:
        """创建数据集统计信息"""
        stats = []
        
        for ticker, metadata in self.stock_metadata.items():
            file_path = self.data_dir / f"{ticker}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                stats.append({
                    'ticker': ticker,
                    'exchange': metadata.exchange,
                    'sector': self.get_sector_from_sic(int(metadata.industry_code)),
                    'market_cap': metadata.market_cap,
                    'start_date': df['date'].min(),
                    'end_date': df['date'].max(),
                    'n_observations': len(df),
                    'avg_price': df['close'].mean(),
                    'avg_volume': df['volume'].mean(),
                })
        
        stats_df = pd.DataFrame(stats)
        
        # 保存统计信息
        stats_path = self.data_dir / 'dataset_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Dataset statistics saved to {stats_path}")
        
        return stats_df
    
    def print_summary(self) -> None:
        """打印数据集摘要"""
        print("=" * 60)
        print("CRSP Large-Scale Dataset Summary")
        print("=" * 60)
        print(f"Time Period: {self.start_year} - {self.end_year}")
        print(f"Total Stocks: {len(self.stock_metadata)}")
        print(f"Data Directory: {self.data_dir}")
        print("\nSector Distribution:")
        for sector, count in sorted(self.get_all_sectors().items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {sector:15s}: {count:5d} stocks")
        print("=" * 60)


class InternationalMarketDataset:
    """
    国际市场数据集（用于迁移学习）
    
    覆盖26个全球市场：
    - 亚洲：中国大陆、香港、日本、印度、韩国、新加坡
    - 欧洲：英国、法国、德国、意大利等18个市场
    - 大洋洲/北美：澳大利亚、新西兰、加拿大
    """
    
    MARKETS = {
        # 亚洲
        'China_A': {'name': '中国大陆', 'source': 'CSMAR', 'currency': 'CNY'},
        'Hong_Kong': {'name': '香港', 'source': 'Datastream', 'currency': 'HKD'},
        'Japan': {'name': '日本', 'source': 'Datastream', 'currency': 'JPY'},
        'India': {'name': '印度', 'source': 'Datastream', 'currency': 'INR'},
        'South_Korea': {'name': '韩国', 'source': 'Datastream', 'currency': 'KRW'},
        'Singapore': {'name': '新加坡', 'source': 'Datastream', 'currency': 'SGD'},
        
        # 欧洲
        'UK': {'name': '英国', 'source': 'Datastream', 'currency': 'GBP'},
        'France': {'name': '法国', 'source': 'Datastream', 'currency': 'EUR'},
        'Germany': {'name': '德国', 'source': 'Datastream', 'currency': 'EUR'},
        'Italy': {'name': '意大利', 'source': 'Datastream', 'currency': 'EUR'},
        'Switzerland': {'name': '瑞士', 'source': 'Datastream', 'currency': 'CHF'},
        'Sweden': {'name': '瑞典', 'source': 'Datastream', 'currency': 'SEK'},
        'Netherlands': {'name': '荷兰', 'source': 'Datastream', 'currency': 'EUR'},
        'Belgium': {'name': '比利时', 'source': 'Datastream', 'currency': 'EUR'},
        'Spain': {'name': '西班牙', 'source': 'Datastream', 'currency': 'EUR'},
        'Austria': {'name': '奥地利', 'source': 'Datastream', 'currency': 'EUR'},
        'Denmark': {'name': '丹麦', 'source': 'Datastream', 'currency': 'DKK'},
        'Finland': {'name': '芬兰', 'source': 'Datastream', 'currency': 'EUR'},
        'Greece': {'name': '希腊', 'source': 'Datastream', 'currency': 'EUR'},
        'Ireland': {'name': '爱尔兰', 'source': 'Datastream', 'currency': 'EUR'},
        'Norway': {'name': '挪威', 'source': 'Datastream', 'currency': 'NOK'},
        'Portugal': {'name': '葡萄牙', 'source': 'Datastream', 'currency': 'EUR'},
        'Russia': {'name': '俄罗斯', 'source': 'Datastream', 'currency': 'RUB'},
        
        # 大洋洲/北美
        'Australia': {'name': '澳大利亚', 'source': 'Datastream', 'currency': 'AUD'},
        'New_Zealand': {'name': '新西兰', 'source': 'Datastream', 'currency': 'NZD'},
        'Canada': {'name': '加拿大', 'source': 'Datastream', 'currency': 'CAD'},
    }
    
    def __init__(self, data_dir: str = "data/international"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_market_data(self, market: str, start_date: str, end_date: str) -> None:
        """
        下载指定市场数据
        
        Note: 实际实现需要根据数据源API调整
        """
        if market not in self.MARKETS:
            raise ValueError(f"Unknown market: {market}")
        
        market_info = self.MARKETS[market]
        logger.info(f"Downloading {market_info['name']} data from {market_info['source']}...")
        
        # TODO: 实现具体的数据下载逻辑
        # 这里需要根据Datastream或CSMAR的API来实现
        
    def load_market_data(self, market: str) -> Optional[pd.DataFrame]:
        """加载指定市场数据"""
        file_path = self.data_dir / f"{market}.csv"
        if file_path.exists():
            return pd.read_csv(file_path, parse_dates=['date'])
        return None


if __name__ == "__main__":
    # 示例用法
    print("CRSP Large-Scale Dataset Loader")
    print("=" * 60)
    
    # 初始化数据集
    dataset = CRSPLargeScaleDataset(
        data_dir="data/crsp_large_scale",
        start_year=1993,
        end_year=2019,
        min_price=1.0,
        min_market_cap=10.0
    )
    
    # 打印摘要（空数据集，仅展示结构）
    dataset.print_summary()
    
    print("\nNote: To download actual data, you need:")
    print("  1. WRDS account with CRSP access")
    print("  2. Run: dataset.download_from_wrds('your_wrds_username')")
    print("\nFor testing without WRDS, use synthetic data generator.")
