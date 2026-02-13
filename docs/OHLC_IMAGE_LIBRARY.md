# OHLC Bar Image Library

基于 Xiu et al. (2021) "(Re-)Imag(in)ing Price Trends" 论文实现的OHLC Bar图像库。

## 图像格式

### 规格

| 参数 | 数值 |
|------|------|
| 背景 | 黑色 (0) |
| 线条 | 白色 (255) |
| Bar宽度 | 3像素 |
| 线条宽度 | 1像素 |

### 图像尺寸

| 时间窗口 | 宽x高 (无Volume) | 宽x高 (有Volume) |
|----------|------------------|------------------|
| 5天 | 15x32 | 15x39 |
| 20天 | 60x64 | 60x77 |
| 60天 | 180x96 | 180x116 |

### OHLC Bar结构

```
    High ─┬─ 垂直线 (粗线，占满bar宽度)
          │
    Open ─┼─← 左水平短线
          │
   Close ─┼─→ 右水平短线
          │
     Low ─┴─ 垂直线
```

- **垂直线**: 连接High和Low，宽度为3像素
- **Open tick**: 左侧水平线，从bar左边缘到中心
- **Close tick**: 右侧水平线，从中心到bar右边缘

## 使用方法

### 1. 快速开始

```bash
# 处理前10只股票（测试用）
python scripts/build_ohlc_image_library.py --limit 10 --window 20 --stride 10

# 处理全部503只标普500股票
python scripts/build_ohlc_image_library.py --all --window 20 --stride 5

# 包含Volume bars
python scripts/build_ohlc_image_library.py --limit 50 --window 20 --volume
```

### 2. 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--window` | 每幅图像的K线数量 | 5, 20, 60 |
| `--stride` | 滑动窗口步长 | 任意整数 |
| `--volume` | 是否包含成交量 | 无（flag） |
| `--limit` | 限制处理的股票数量 | 整数 |
| `--all` | 处理全部503只股票 | 无（flag） |

### 3. Python API

```python
from core.utils.ohlc_renderer import OHLCRenderer, ImageLibrary
from core.tradingview_fetcher import TradingViewFetcher

# 直接渲染
fetcher = TradingViewFetcher()
df = fetcher.fetch_ohlcv("AAPL", n_bars=20)

renderer = OHLCRenderer(n_bars=20, include_volume=False)
img = renderer.render(df)
img.save("aapl_ohlc.png")

# 建立图像库
library = ImageLibrary(
    library_dir="data/image_library",
    n_bars=20,
    include_volume=False
)

# 添加股票数据
result = library.add_stock_data(
    symbol="AAPL",
    df=df,
    stride=5
)
print(f"Generated {result['num_images']} images")
```

## 目录结构

```
data/image_library/
├── processed/
│   ├── MMM/
│   │   ├── MMM_19900102_19900129_000000.png
│   │   ├── MMM_19900102_19900129_000010.png
│   │   └── ...
│   ├── AAPL/
│   │   └── ...
│   └── ...
├── metadata.csv
└── summary.txt
```

## 元数据格式

`metadata.csv` 包含以下字段：

| 字段 | 说明 |
|------|------|
| symbol | 股票代码 |
| filename | 图像文件名 |
| filepath | 相对路径 |
| window_start | 窗口起始索引 |
| window_end | 窗口结束索引 |
| start_date | 起始日期 |
| end_date | 结束日期 |
| open_price | 开盘价 |
| close_price | 收盘价 |
| return | 窗口收益率 |

## 示例图像

### 20天OHLC Bar (60x64)

![Example](test_xiu_lib/processed/AAPL/AAPL_20210104_20210201_000000.png)

- 黑色背景，白色线条
- 20根OHLC bars
- 每个bar宽3像素
- 文件大小约300-400字节（高度稀疏）

## 参考

Xiu, D., Huang, J., & Zhang, Y. (2021). (Re-)Imag(in)ing Price Trends. 
*The Journal of Finance*.
