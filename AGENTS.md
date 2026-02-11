# K-Line Visual Pattern Search - AI Agent Guide

> 本文档面向 AI 编程助手，提供项目架构、开发流程和技术栈的完整指南。

## 项目概述

本项目是一个基于深度学习的 **K线形态相似性搜索与股价预测系统**。核心功能包括：

- **K线图像生成**：将OHLCV数据渲染为标准化图像（蜡烛图、OHLC条形图、GAF编码）
- **特征提取**：使用CNN/ResNet提取K线形态的向量表示
- **相似性搜索**：基于FAISS向量数据库的快速形态匹配
- **股价预测**：CNN和Transformer模型预测未来价格走势（涨跌二分类）

### 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.0+, torchvision |
| 后端API | FastAPI, Uvicorn |
| 图像处理 | OpenCV (headless), PIL, matplotlib, mplfinance |
| 向量搜索 | FAISS-CPU |
| 数据处理 | NumPy, Pandas, PyArrow, Scikit-learn |
| 金融数据 | yfinance, akshare |
| 时序分析 | dtw-python, pyts |

---

## 项目结构

```
k-line-analyze/
├── core/                          # 核心功能模块
│   ├── config.py                 # 配置管理 (pydantic-settings)
│   ├── data_fetcher.py           # 股票数据获取 (US/TW/CN/HK/Crypto)
│   ├── dataset.py                # 数据集定义
│   ├── preprocessor.py           # 数据预处理
│   ├── similarity_search.py      # 相似性搜索 (FAISS)
│   ├── dtw_matcher.py            # DTW动态时间规整匹配
│   ├── feature_extractor.py      # 特征提取服务
│   └── utils/                    # 工具函数
│       ├── kline_renderer.py     # K线渲染
│       └── gaf_transformer.py    # GAF图像变换
│
├── models/                        # 模型定义
│   ├── cnn_encoder.py            # CNN编码器 (主模型, ResNet18)
│   ├── cnn_predictor.py          # CNN预测器 (v1, 60天/128px)
│   ├── cnn_predictor_v2.py       # CNN预测器 (v2, 10天/256px)
│   ├── transformer_predictor.py  # Transformer预测器
│   ├── prediction_dataset.py     # 预测数据集 (v1)
│   ├── prediction_dataset_v2.py  # 预测数据集 (v2)
│   └── baselines.py              # 基线模型 (LSTM, ResNet1D)
│
├── src/                           # 扩展模块
│   ├── data/
│   │   ├── data_loader.py        # 高级数据加载
│   │   ├── dataset.py            # 高级数据集
│   │   ├── image_generator.py    # 图像生成器 (OHLC/GAF/混合)
│   │   ├── multiscale_dataset.py # 多尺度数据集
│   │   ├── quantile_filtering.py # 分位数筛选（训练集专用，防泄漏）
│   │   └── advanced_labeling.py  # 高级标签策略
│   └── models/
│       ├── cnn_model.py          # CNN模型变体
│       ├── lightweight_cnn.py    # 轻量CNN（RTX 4060优化）
│       ├── rnn_model.py          # RNN模型
│       ├── multiscale_cnn.py     # 多尺度CNN
│       ├── vision_transformer.py # ViT模型
│       └── transformer_model.py  # Transformer变体
│
├── backend/                       # FastAPI后端服务
│   ├── app/
│   │   ├── main.py               # FastAPI应用入口
│   │   ├── api/routes/           # API路由
│   │   │   ├── prediction.py     # v1预测API
│   │   │   └── prediction_v2.py  # v2预测API
│   │   ├── services/             # 业务服务
│   │   │   ├── feature_extractor.py
│   │   │   ├── data_fetcher.py
│   │   │   └── preprocessor.py
│   │   └── models/               # 后端模型副本
│   └── trained_models/           # 训练好的模型存储
│
├── scripts/                       # 训练和实验脚本
│   ├── download_*.py             # 数据下载脚本
│   ├── train_*_predictor*.py     # 模型训练脚本
│   ├── train_grouped_4060.py     # RTX 4060分组训练（按板块）
│   ├── train_3class_predictor.py # 3分类训练（涨/中性/跌）
│   ├── test_4060_setup.py        # RTX 4060环境测试
│   ├── run_*_experiments.py      # 实验运行脚本
│   └── visualize_*.py            # 可视化工具
│
├── data/                          # 数据目录
│   └── raw/                      # 原始股票数据 (CSV)
│       ├── us/                   # 美股数据
│       └── cn/                   # A股数据
│
├── outputs/                       # 输出目录
│   ├── models/                   # 保存的模型权重
│   └── indices/                  # FAISS索引文件
│
├── docs/                          # 文档
│   ├── prediction-models.md      # 预测模型使用说明
│   ├── RTX4060_TRAINING.md       # RTX 4060训练指南
│   ├── 3CLASS_NEUTRAL.md         # 3分类（含中性）说明
│   ├── Baseline_EXPERIMENTS.md       # Baseline实验记录
│   ├── V2_IMPROVEMENTS.md        # V2改进文档
│   └── V2_COMPLETE_SUMMARY.md    # V2完整总结
│
├── requirements.txt               # Python依赖
└── run_experiments.bat           # Windows批处理脚本
```

---

## 关键配置

### 核心参数 (core/config.py)

```python
# 模型设置 (针对RTX 4060 8GB VRAM优化)
EMBEDDING_DIM = 512              # 嵌入向量维度
IMAGE_SIZE = 256                 # 图像分辨率
DEFAULT_WINDOW_SIZES = [10, 20, 30, 40, 60, 90, 120, 180, 240]  # 多尺度窗口

# 搜索设置
DEFAULT_TOP_K = 50               # 相似搜索返回数量
FAISS_NPROBE = 512               # FAISS搜索精度

# LLM API设置 (可选)
LLM_PROVIDER = "gemini"          # gemini 或 qwen
LLM_MODEL = "gemini-2.0-flash"
LLM_ENABLED = false
```

### 环境变量 (.env)

```bash
# LLM API密钥 (可选)
LLM_API_KEY=your_api_key_here
LLM_ENABLED=true

# 其他配置
DEBUG=true
```

---

## 构建与运行

### 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: OpenCV使用headless版本 (`opencv-python-headless`) 以支持服务器/容器环境。

### 启动后端服务

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API文档地址: http://localhost:8000/docs

### 运行实验

```bash
# 运行完整Baseline实验
python scripts/run_baseline_experiments.py

# 运行V2实验
python scripts/run_v2_experiments.py --experiment all --num-runs 3

# 训练CNN v2预测模型 (推荐)
python scripts/train_cnn_predictor_v2.py --mode all --epochs 50

# 训练Transformer v2预测模型
python scripts/train_transformer_predictor_v2.py --mode all --epochs 80

# 运行优化流水线 (6阶段)
python scripts/run_optimized_pipeline.py --stage all
```

---

## 数据格式

### CSV文件格式

股票数据存储为CSV文件，包含以下列：

```csv
date,open,high,low,close,volume
2024-01-01,150.0,155.0,149.0,154.0,1000000
...
```

数据存储位置:
- 美股: `data/raw/us/{SYMBOL}.csv`
- A股: `data/raw/cn/{SYMBOL}.csv`

### 图像表示方法

| 方法 | 说明 | 分辨率 | 论文参考 |
|------|------|--------|----------|
| **Candle** | 传统蜡烛图 | 128×128 或 256×256 | - |
| **OHLC Bars** | 条形图 (Xiu et al.) | 256×256 | (Re-)Imag(in)ing Price Trends |
| **GAF** | Gramian Angular Field | 256×256 | Chen & Tsai (2020) |
| **Hybrid** | OHLC + GAF混合 | 256×256 | - |

### v1 vs v2 配置对比

| 项目 | v1 | v2 (推荐) |
|------|-----|----------|
| 窗口大小 | 60天 | 10天 |
| 图像尺寸 | 128×128 | 256×256 |
| 每根K线宽度 | ~2px | ~25px |
| 预测类别 | 多分类 | 二分类 (涨/跌) |

---

## 代码规范

### 命名约定

- **文件**: 小写+下划线 (e.g., `cnn_encoder.py`)
- **类**: PascalCase (e.g., `KLineEncoder`)
- **函数/变量**: snake_case (e.g., `extract_features`)
- **常量**: UPPER_CASE (e.g., `EMBEDDING_DIM`)

### 注释风格

使用Google风格的docstring:

```python
def process_data(data: pd.DataFrame, window_size: int = 60) -> np.ndarray:
    """
    处理股票数据为模型输入。
    
    Args:
        data: OHLCV DataFrame
        window_size: 时间窗口大小
        
    Returns:
        处理后的numpy数组
        
    Raises:
        ValueError: 数据格式不正确时
    """
```

### 类型注解

重要函数必须添加类型注解:

```python
from typing import Tuple, Optional, List, Dict

def encode(self, x: torch.Tensor) -> np.ndarray:
    ...
```

---

## 模型架构

### CNN编码器 (KLineEncoder)

基于ResNet18修改:
- 输入: (batch, 3, H, W) RGB图像
- 输出: (batch, embedding_dim) L2归一化向量
- 包含Embedding Head和Prediction Head双输出

```python
class KLineEncoder(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # 移除分类头
        self.embedding_head = nn.Sequential(...)
        self.prediction_head = nn.Sequential(...)
```

### CNN预测器 v2 (KLineCNNPredictorV2)

```python
class KLineCNNPredictorV2(nn.Module):
    """
    输入: (batch, 3, 256, 256) - 10天K线图
    输出: dict - 每个horizon的分类和回归结果
    """
    def __init__(self, num_classes=2, predict_horizons=[1, 3, 5]):
        self.backbone = resnet18(pretrained=True)
        # 每个horizon独立的分类头和回归头
        self.cls_heads = nn.ModuleDict()
        self.reg_heads = nn.ModuleDict()
```

### Transformer预测器

```python
class KLineTransformerPredictor(nn.Module):
    """
    输入: (batch, seq_len, n_features) - 9维特征序列
    输出: (batch, num_classes) - 二分类概率
    """
    def __init__(self, n_features=9, d_model=128, nhead=8, num_encoder_layers=4):
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(...)
```

### 损失函数

- **TripletLoss**: 自监督学习的三元组损失
- **CombinedLoss**: Triplet + Prediction 多任务损失
- **TransformerPredictionLoss**: 分类 + 回归联合损失
- **CrossEntropyLoss**: 分类任务

---

## API端点

### 预测API (v2)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/prediction/v2/models` | GET | 列出所有v2模型 |
| `/api/v1/prediction/v2/predict` | POST | 股价预测 |
| `/api/v1/prediction/v2/compare/{symbol}` | GET | 模型对比 |

### 预测请求示例

```bash
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "market": "us",
    "model_type": "cnn",
    "model_variant": "universal"
  }'
```

### 响应格式

```json
{
  "symbol": "AAPL",
  "market": "us",
  "model_type": "transformer",
  "window_size": 10,
  "image_size": 256,
  "predictions": {
    "T+1": {
      "predicted_class": "涨",
      "class_probabilities": {"跌": 0.35, "涨": 0.65},
      "predicted_return": 0.008
    },
    "T+3": {...},
    "T+5": {...}
  }
}
```

---

## 训练策略

### 数据划分

**重要**: 使用时间切分而非随机切分！

```python
# 错误 (数据泄漏)
train_test_split(X, y, test_size=0.2, random_state=42)

# 正确 (按时间顺序)
from models.prediction_dataset_v2 import time_based_split
train_subset, val_subset = time_based_split(dataset, val_ratio=0.15)
```

### 数据泄漏防护

**关键原则**: 筛选/过滤只应用于训练集，验证集和测试集必须保持完整分布！

```python
# ❌ 错误: 所有集合都筛选（导致数据泄漏）
common = {'quantile_filter': 0.35}
train_ds = StockDataset(mode='train', **common)
val_ds = StockDataset(mode='val', **common)    # 错误！
test_ds = StockDataset(mode='test', **common)  # 错误！

# ✅ 正确: 只筛选训练集
common = {}
train_ds = StockDataset(
    mode='train', 
    quantile_filter=0.35,      # 只用于训练
    train_filter_threshold=0.003,  # 只用于训练
    **common
)
val_ds = StockDataset(mode='val', **common)    # 无筛选
test_ds = StockDataset(mode='test', **common)  # 无筛选
```

### 训练模式

1. **Universal (全体模型)**: 使用所有股票数据训练，泛化能力强
2. **Single (单一股票)**: 针对特定股票训练，捕捉个股特有模式  
3. **Grouped (分组模型)**: 按产业板块分别训练，捕捉行业特征

### 优化技巧

- **混合精度训练**: 使用 `torch.cuda.amp` 加速
- **模型编译**: PyTorch 2.0+ 使用 `torch.compile()`
- **数据加载**: 多worker + pin_memory + persistent_workers
- **早停**: patience=5-7，防止过拟合
- **学习率调度**: CosineAnnealingWarmRestarts
- **梯度裁剪**: max_norm=1.0

### GPU优化配置

```python
# 环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 修复OpenMP冲突

# PyTorch设置
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

---

## 内置股票列表

### 美股 (26只)

```
AAPL  MSFT  GOOGL  AMZN  NVDA  TSLA  META  AMD
NFLX  INTC  JPM    BAC   V     MA    JNJ   PFE
UNH   PG    KO     WMT   XOM   CVX   BA    DIS
CRM   CSCO
```

覆盖科技、金融、医疗、消费、能源五大板块。

---

## 文件生成位置

### 模型输出

```
backend/trained_models/predictors/
├── cnn_v2_universal.pt              # CNN全体模型
├── cnn_v2_us_AAPL.pt               # CNN单一股票模型
├── transformer_v2_universal.pt      # Transformer全体模型
├── *_history.json                   # 训练历史
└── *_config.json                    # 模型配置
```

### 分组模型输出 (RTX 4060)

```
outputs/grouped_models_4060/
├── Tech_Software_3class.pt              # 科技-软件板块
├── Tech_Semiconductors_3class.pt        # 科技-半导体板块
├── Financials_3class.pt                 # 金融板块
├── Healthcare_3class.pt                 # 医疗板块
├── Consumer_3class.pt                   # 消费品板块
├── Industrials_Energy_3class.pt         # 工业能源板块
└── summary.json                         # 汇总结果
```

### 实验结果

```
outputs/baseline_results/
├── intermediate_YYYYMMDD_HHMMSS.json    # 中间结果
└── full_results_YYYYMMDD_HHMMSS.json    # 完整结果

outputs/3class_models/                   # 3分类模型
└── 3class_model_YYYYMMDD_HHMMSS.pt

outputs/optimized_pipeline/              # 优化流水线各阶段
├── stage1/                              # 基线优化
├── stage2/                              # 多架构训练
├── stage3/                              # 多尺度融合
├── stage4/                              # 集成评估
├── stage5/                              # 分组微调
├── stage6/                              # 技术指标
└── stage7/                              # 3分类实验
```

---

## 常见问题

### OpenMP冲突

```python
# 在导入torch/numpy前设置
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### 数据不足错误

确保股票数据文件存在且包含足够的历史数据:
```
data/raw/us/AAPL.csv  (至少包含 window_size * 3 天的数据)
```

### 模型加载失败

检查模型路径是否正确，或使用universal模型作为fallback:
```python
if not model_path.exists():
    model_path = models_dir / f"{model_type}_v2_universal.pt"
```

---

## 硬件配置指南

### RTX 4060 8GB 配置

专为消费级显卡优化的轻量配置：

```python
# 轻量级CNN (50万参数 vs ResNet18的1100万)
from src.models.lightweight_cnn import build_lightweight_cnn
model = build_lightweight_cnn(variant='light', num_classes=3)

# 推荐训练参数
CONFIG = {
    'batch_size': 32,           # 8GB显存安全值
    'num_workers': 4,           # 根据CPU核心调整
    'window_size': 20,          # 20天回顾
    'prediction_horizon': 5,    # 预测5天
    'img_size': (128, 128),     # 输入尺寸
    'epochs': 50,
    'lr': 1e-3,
}
```

训练命令：
```bash
# 测试环境
python scripts/test_4060_setup.py

# 分组训练（6大板块）
python scripts/train_grouped_4060.py --all

# 3分类模式
python scripts/train_grouped_4060.py --all --num-classes 3
```

### H20 96GB 配置

大显存服务器的训练配置：

```python
CONFIG = {
    'batch_size': 128,      # H20大显存可用更大batch
    'num_workers': 8,       # 32核CPU
    'pin_memory': True,
    'prefetch_factor': 4,
    'use_amp': True,        # 混合精度
    'use_compile': True,    # torch.compile
}
```

---

## 扩展开发

### 添加新的图像表示

在 `src/data/image_generator.py` 中添加新方法:

```python
def draw_new_chart(self, open_p, high_p, low_p, close_p, ...):
    """新的K线表示方法"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # 绘制逻辑...
    return img
```

### 添加新的预测模型

1. 在 `models/` 创建模型类继承 `nn.Module`
2. 实现 `forward()`, `save()`, `load()` 方法
3. 在 `backend/app/models/` 创建服务包装
4. 在 `scripts/` 创建训练脚本
5. 在 `backend/app/api/routes/` 添加API路由

---

## 参考文献

1. **Xiu et al. (2021)** - "(Re-)Imag(in)ing Price Trends"
2. **Chen & Tsai (2020)** - "Encoding candlesticks as images"
3. **Duong et al. (2025)** - "Investigating Market Strength Prediction"

---

## 许可证

MIT License
