# TradingView 美股数据源

本项目已集成 TradingView 作为美股数据的主要数据源，覆盖全部**标普500指数成分股（503只）**。

## 特点

- **完整覆盖**: 全部标普500成分股（503只）
- **更长历史**: 可获取1990年至今的数据
- **更高质量**: TradingView的数据质量优于Yahoo Finance
- **多种周期**: 支持1分钟到月线的多种时间周期
- **自动回退**: TradingView失败时自动回退到Yahoo Finance
- **本地缓存**: 支持增量更新，避免重复下载

## 安装

```bash
pip install tradingview-datafeed websocket-client
```

或

```bash
pip install -r requirements.txt
```

## 快速使用

### 1. 命令行工具

```bash
# 下载全部标普500股票
python scripts/download_all_us_stocks.py

# 只下载前10只（测试用）
python scripts/download_all_us_stocks.py --limit 10

# 下载特定股票
python scripts/download_all_us_stocks.py --symbols AAPL MSFT GOOGL

# 强制重新下载
python scripts/download_all_us_stocks.py --force

# 查看数据库信息
python scripts/download_all_us_stocks.py --info
```

### 2. Python API

```python
from core.tradingview_fetcher import TradingViewFetcher, USStockDatabase

# 直接获取数据
fetcher = TradingViewFetcher()
df = fetcher.fetch_ohlcv("AAPL", n_bars=5000)

# 使用本地数据库
db = USStockDatabase(data_dir="data/raw/us")
db.update_stock("AAPL")  # 增量更新
df = db.load_stock("AAPL")
```

### 3. 集成到 DataFetcher

```python
from core.data_fetcher import DataFetcher

# 自动使用 TradingView 作为美股数据源
fetcher = DataFetcher(use_tradingview=True)
df = await fetcher.fetch_ohlcv("AAPL", market="us")
```

## 标普500成分股分布

| 交易所 | 股票数量 | 占比 |
|--------|----------|------|
| NASDAQ | 84 | 16.7% |
| NYSE | 419 | 83.3% |
| **总计** | **503** | **100%** |

### GICS板块分布

| 板块 | 大约数量 | 代表性股票 |
|------|----------|------------|
| Information Technology | ~70 | AAPL, MSFT, NVDA, AMD |
| Health Care | ~65 | JNJ, UNH, PFE, ABBV |
| Financials | ~65 | JPM, BAC, GS, MS |
| Consumer Discretionary | ~55 | AMZN, TSLA, HD, NKE |
| Industrials | ~70 | HON, UPS, BA, CAT |
| Communication Services | ~25 | GOOGL, META, NFLX, DIS |
| Consumer Staples | ~35 | WMT, PG, KO, PEP |
| Energy | ~25 | XOM, CVX, COP, SLB |
| Utilities | ~30 | NEE, DUK, SO, D |
| Real Estate | ~30 | AMT, PLD, CCI, EQIX |
| Materials | ~25 | LIN, SHW, NEM, FCX |

## 数据格式

数据以 Parquet 格式存储在 `data/raw/us/` 目录：

```
data/raw/us/
├── MMM.parquet
├── AOS.parquet
├── ABT.parquet
├── ABBV.parquet
├── ACN.parquet
└── ... (503 files total)
```

每个文件包含以下列：
- `symbol`: 交易所:股票代码 (如 NASDAQ:AAPL)
- `Open`: 开盘价
- `High`: 最高价
- `Low`: 最低价
- `Close`: 收盘价
- `Volume`: 成交量

## 更新股票列表

如果需要更新标普500成分股列表（通常每季度更新）：

```bash
# 获取最新列表并更新代码
python scripts/fetch_sp500_list.py

# 修复交易所映射
python scripts/fix_exchange_mapping.py
```

## 注意事项

1. **请求频率**: 默认每只股票之间有0.5秒延迟，避免请求过快
2. **数据限制**: 无账号情况下单次最多获取5000根K线
3. **网络连接**: TradingView连接偶尔会断开，会自动重连
4. **数据存储**: 建议使用SSD存储，503只股票的完整数据约需500MB-1GB空间

## 测试

```bash
python scripts/test_tradingview.py
```

## 故障排除

### 连接错误

如果出现 `Connection to remote host was lost`，这是正常现象，库会自动重连。

### 无数据返回

检查 symbol 和交易所映射是否正确：
```python
from core.tradingview_fetcher import EXCHANGE_MAP
print(EXCHANGE_MAP.get('YOUR_SYMBOL'))  # 应输出 NASDAQ 或 NYSE
```

### 编码错误

Windows下如果出现编码问题，请设置环境变量：
```bash
chcp 65001
```

## 数据来源

- **标普500成分股列表**: [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies)
- **股票价格数据**: [TradingView](https://www.tradingview.com/) (via tvdatafeed)
