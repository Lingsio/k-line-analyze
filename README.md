# K-Line Visual Pattern Search

基于深度学习的K线形态相似性搜索工具。将K线（蜡烛图）转换为可搜索的向量嵌入，实现快速的形态相似性检索。

## 核心功能

- **K线图像生成**: 将OHLCV数据渲染为标准化的K线图像
- **特征提取**: 使用CNN/ResNet提取K线形态的向量表示
- **相似性搜索**: 基于向量距离的快速形态匹配
- **DTW匹配**: 动态时间规整的序列相似度计算

## 安装

```bash
pip install -r requirements.txt
```

**依赖**: PyTorch, torchvision, numpy, pandas, scikit-learn, matplotlib, mplfinance

## 项目结构

```
k-line-photo/
├── core/                    # 核心功能模块
│   ├── config.py           # 配置管理
│   ├── data_fetcher.py     # 数据获取
│   ├── dataset.py          # 数据集定义
│   ├── preprocessor.py     # 数据预处理
│   ├── similarity_search.py # 相似性搜索
│   ├── dtw_matcher.py      # DTW匹配
│   ├── feature_extractor.py # 特征提取
│   └── utils/              # 工具函数
│       ├── kline_renderer.py  # K线渲染
│       └── gaf_transformer.py # GAF变换
├── models/                  # 模型定义
│   ├── cnn_encoder.py      # CNN编码器（主模型）
│   └── baselines.py        # 基线模型（LSTM, ResNet1D）
├── src/                     # 扩展模块
│   ├── data/               # 数据处理
│   │   ├── data_loader.py  # 数据加载
│   │   ├── dataset.py      # 高级数据集
│   │   └── image_generator.py # 图像生成器
│   └── models/             # 模型变体
│       ├── cnn_model.py    # CNN模型
│       └── rnn_model.py    # RNN模型
├── scripts/                 # 数据下载脚本
│   ├── download_cn_stocks.py    # 下载A股数据
│   ├── download_cn_baostock.py  # BaoStock数据源
│   └── download_us_stocks.py    # 下载美股数据
└── data/raw/               # 原始数据存储
```

## 快速开始

### 1. 下载数据

```bash
# 下载A股数据
python scripts/download_cn_baostock.py

# 下载美股数据
python scripts/download_us_stocks.py
```

### 2. 基本使用

```python
from core.preprocessor import Preprocessor
from core.dataset import KLineDataset
from models.cnn_encoder import CNNEncoder

# 加载数据
preprocessor = Preprocessor()
data = preprocessor.load_stock_data("data/raw/AAPL.csv")

# 创建数据集
dataset = KLineDataset(data, window_size=60)

# 加载模型
model = CNNEncoder(embedding_dim=256)
```

### 3. 生成K线图像

```python
from src.data.image_generator import ImageGenerator

generator = ImageGenerator(
    figsize=(128, 128),
    normalization='minmax'
)
image = generator.generate(ohlcv_data)
```

### 4. 相似性搜索

```python
from core.similarity_search import SimilaritySearch

searcher = SimilaritySearch(model)
similar_patterns = searcher.search(query_pattern, k=10)
```

## 数据格式

CSV文件包含以下列：
- `date`: 日期
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `WINDOW_SIZE` | 60 | 输入天数 |
| `IMAGE_SIZE` | 128 | 图像分辨率 |
| `EMBEDDING_DIM` | 256 | 嵌入向量维度 |
| `BATCH_SIZE` | 32 | 批次大小 |

## 文档导航

### 方法文档 (METHOD_)

| 文档 | 内容 |
|------|------|
| `THEORY.md` | 完整理论框架与数学原理 |
| `METHOD_IMAGE_ENCODING.md` | 图像编码方法 (OHLC/GAF/Hybrid) |
| `METHOD_MULTISCALE_FUSION.md` | 多尺度 CNN 融合方法 |
| `METHOD_OPTIMIZED_PIPELINE.md` | 6 阶段优化流水线指南 |
| `docs/METHOD_3CLASS_CLASSIFICATION.md` | 三分类方法 (涨/平/跌) |
| `docs/METHOD_RTX4060_TRAINING.md` | RTX 4060 训练指南 |

### 实验结果 (RESULTS_)

| 文档 | 内容 |
|------|------|
| `RESULTS_BASELINE_EXPERIMENTS.md` | 基线实验对比 (CNN/LSTM/ResNet1D) |
| `docs/RESULTS_EXPERIMENTS_COMPARISON.md` | 全量实验对比总结 |
| `docs/RESULTS_SAK_NET_2026-02-11.md` | **SAK-Net 实验 (最新成果 57.81%)** |
| `docs/RESULTS_GROUPED_TRAINING.md` | 分组训练方法详解 |
| `docs/RESULTS_PER_STOCK_TRAINING.md` | 单股票训练实验 |

### 用户指南

| 文档 | 内容 |
|------|------|
| `README_PROJECT.md` | 完整项目介绍与快速开始 |
| `docs/USER_GUIDE_PREDICTION_MODELS.md` | 预测模型 API 使用说明 |

## License

MIT License
