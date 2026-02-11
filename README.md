# K-Line Visual Pattern Recognition System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**语言 / Language**: [中文](#中文) | [English](#english)

---

<a id="中文"></a>
# 中文版本

## 项目简介

本项目是一个**基于深度学习的 K 线形态相似性搜索与股价预测系统**。核心创新在于将股票 K 线数据转换为图像，利用 CNN 的图像识别能力自动学习价格模式，实现：

- **🔍 相似形态搜索**: 基于 FAISS 向量数据库快速匹配历史相似 K 线形态
- **📈 股价趋势预测**: CNN/Transformer 模型预测未来涨跌方向
- **🏭 分组专业模型**: 按行业板块训练专门模型，显著提升准确率

### 核心成果 (SAK-Net)

| 指标 | 结果 |
|------|------|
| **平均准确率** | **57.81%** (目标 54%) |
| **最佳板块** | Consumer (60.37%) |
| **覆盖股票** | 57 只美股 |
| **行业板块** | 6 大板块 |

#### SAK-Net 分组实验结果

| 板块 | 股票数 | 准确率 | F1 | 状态 |
|------|--------|--------|-----|------|
| **Consumer** 🏆 | 10 | **60.37%** | 0.59 | ✅ 推荐部署 |
| **Industrials_Energy** | 8 | **59.94%** | 0.52 | ✅ 推荐部署 |
| **Tech_Semiconductors** | 6 | **59.88%** | 0.57 | ✅ 推荐部署 |
| **Tech_Software** | 14 | **57.22%** | 0.58 | ✅ 可部署 |
| **Financials** | 9 | **55.90%** | 0.55 | ⚠️ 需优化 |
| **Healthcare** | 10 | **53.54%** | 0.52 | ⚠️ 需优化 |
| **平均** | **57** | **57.81%** | - | ✅ 超目标 |

---

## 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-org/k-line-analyze.git
cd k-line-analyze

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
# 下载美股数据
python scripts/download_us_stocks.py

# 数据存储位置
data/raw/us/
├── AAPL.csv
├── MSFT.csv
└── ...
```

### 训练模型

#### 1. 分组训练 (推荐)

```bash
# 训练所有板块 (6 个分组模型)
python scripts/train_grouped_h20.py --all --num-runs 3

# 训练指定板块
python scripts/train_grouped_h20.py --sector Consumer
```

#### 2. 通用模型

```bash
# CNN v2 (图像路线)
python scripts/train_cnn_predictor_v2.py --mode all --epochs 50

# Transformer v2 (序列路线)
python scripts/train_transformer_predictor_v2.py --mode all --epochs 80

# 单一股票模型
python scripts/train_cnn_predictor_v2.py --mode single --symbol AAPL
```

### 启动 API 服务

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 API 文档。

---

## 使用说明

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/prediction/v2/models` | GET | 列出所有已训练的 v2 模型 |
| `/api/v1/prediction/v2/predict` | POST | 使用指定模型进行预测 |
| `/api/v1/prediction/v2/compare/{symbol}` | GET | 对比所有 v2 模型的预测结果 |

### 预测请求示例

```bash
# CNN 全体模型预测 AAPL
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "market": "us", "model_type": "cnn"}'

# CNN 单一股票模型预测 AAPL
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "market": "us", "model_type": "cnn", "model_variant": "cnn_v2_us_AAPL"}'

# 对比所有模型
curl http://localhost:8000/api/v1/prediction/v2/compare/TSLA?market=us
```

### 返回格式

```json
{
  "symbol": "AAPL",
  "market": "us",
  "model_type": "cnn",
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

## 项目结构

```
k-line-analyze/
├── core/                          # 核心功能模块
│   ├── config.py                 # 配置管理
│   ├── data_fetcher.py           # 股票数据获取
│   ├── similarity_search.py      # FAISS 相似性搜索
│   ├── feature_extractor.py      # 特征提取
│   └── utils/                    # 工具函数
│       ├── kline_renderer.py     # K线渲染
│       └── gaf_transformer.py    # GAF变换
│
├── models/                        # 模型定义
│   ├── cnn_encoder.py            # CNN 编码器
│   ├── cnn_predictor_v2.py       # CNN 预测器 v2
│   ├── transformer_predictor.py  # Transformer 预测器
│   └── prediction_dataset_v2.py  # 预测数据集
│
├── src/                           # 扩展模块
│   ├── data/
│   │   ├── image_generator.py    # 图像生成器 (OHLC/GAF/混合)
│   │   └── dataset.py            # 高级数据集
│   └── models/
│       ├── cnn_model.py          # CNN 模型变体
│       └── lightweight_cnn.py    # 轻量 CNN
│
├── backend/                       # FastAPI 后端服务
│   └── app/
│       ├── main.py               # API 入口
│       ├── api/routes/           # API 路由
│       └── services/             # 业务服务
│
├── scripts/                       # 训练和实验脚本
│   ├── train_grouped_h20.py      # 分组训练 (H20)
│   ├── train_cnn_predictor_v2.py # CNN v2 训练
│   └── download_us_stocks.py     # 数据下载
│
├── data/                          # 数据目录
│   └── raw/us/                   # 美股数据
│
├── outputs/                       # 输出目录
│   └── models/                   # 保存的模型权重
│
└── docs/                          # 文档
```

---

## 核心技术

### 图像编码方法

| 方法 | 描述 | 论文参考 |
|------|------|----------|
| **Candlestick** | 传统蜡烛图 | - |
| **OHLC Bars** | 稀疏条形图 (推荐) | Xiu et al. (2021) |
| **GAF** | Gramian Angular Field | Chen & Tsai (2020) |
| **Hybrid** | 多通道融合 | - |

### 模型架构

| 架构 | 输入 | 特点 |
|------|------|------|
| **CNN (ResNet18)** | K 线图 256×256 | 图像识别，预训练权重 |

| **分组模型** | 按板块训练 | 专业优化，异质处理 |

### 关键发现

1. **分组训练** 是最有效的提升手段 (+7%)
2. **OHLC 编码** 优于传统蜡烛图
3. **20天窗口** + **128px** 是最佳平衡
4. **ResNet18** + **预训练权重** 是关键

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [THEORY.md](./THEORY.md) | 完整理论框架、数据编码方法、模型架构、训练方法 |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | 完整实验报告：SAK-Net (57.81%)、基线对比、分组训练、失败分析 |

---

## License

MIT License - 详见 [LICENSE](LICENSE) 文件

> ⚠️ **免责声明**: 本项目仅供学术研究使用，不构成投资建议。股市有风险，投资需谨慎。

---

<a id="english"></a>
# English Version

## Project Introduction

This project is a **deep learning-based K-line pattern similarity search and stock price prediction system**. The core innovation lies in converting stock K-line data into images and leveraging CNN image recognition capabilities to automatically learn price patterns, achieving:

- **🔍 Similar Pattern Search**: Fast matching of historical similar K-line patterns based on FAISS vector database
- **📈 Stock Trend Prediction**: CNN/Transformer models for predicting future price direction
- **🏭 Sector-Specific Models**: Specialized models trained by industry sector, significantly improving accuracy

### Core Results (SAK-Net)

| Metric | Result |
|------|------|
| **Average Accuracy** | **57.81%** (Target: 54%) |
| **Best Sector** | Consumer (60.37%) |
| **Stocks Covered** | 57 US stocks |
| **Industry Sectors** | 6 major sectors |

#### SAK-Net Grouped Training Results

| Sector | Stocks | Accuracy | F1 | Status |
|------|--------|--------|-----|------|
| **Consumer** 🏆 | 10 | **60.37%** | 0.59 | ✅ Recommended |
| **Industrials_Energy** | 8 | **59.94%** | 0.52 | ✅ Recommended |
| **Tech_Semiconductors** | 6 | **59.88%** | 0.57 | ✅ Recommended |
| **Tech_Software** | 14 | **57.22%** | 0.58 | ✅ Deployable |
| **Financials** | 9 | **55.90%** | 0.55 | ⚠️ Needs optimization |
| **Healthcare** | 10 | **53.54%** | 0.52 | ⚠️ Needs optimization |
| **Average** | **57** | **57.81%** | - | ✅ Above target |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/k-line-analyze.git
cd k-line-analyze

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download US stock data
python scripts/download_us_stocks.py

# Data storage location
data/raw/us/
├── AAPL.csv
├── MSFT.csv
└── ...
```

### Model Training

#### 1. Grouped Training (Recommended)

```bash
# Train all sectors (6 grouped models)
python scripts/train_grouped_h20.py --all --num-runs 3

# Train specific sector
python scripts/train_grouped_h20.py --sector Consumer
```

#### 2. Universal Models

```bash
# CNN v2 (image approach)
python scripts/train_cnn_predictor_v2.py --mode all --epochs 50

# Transformer v2 (sequence approach)
python scripts/train_transformer_predictor_v2.py --mode all --epochs 80

# Single-stock model
python scripts/train_cnn_predictor_v2.py --mode single --symbol AAPL
```

### Launch API Service

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for API documentation.

---

## Usage Guide

### API Endpoints

| Endpoint | Method | Description |
|------|------|------|
| `/api/v1/prediction/v2/models` | GET | List all trained v2 models |
| `/api/v1/prediction/v2/predict` | POST | Predict using specified model |
| `/api/v1/prediction/v2/compare/{symbol}` | GET | Compare predictions from all models |

### Prediction Request Examples

```bash
# CNN universal model predicting AAPL
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "market": "us", "model_type": "cnn"}'

# CNN single stock model predicting AAPL
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "market": "us", "model_type": "cnn", "model_variant": "cnn_v2_us_AAPL"}'

# Compare all models
curl http://localhost:8000/api/v1/prediction/v2/compare/TSLA?market=us
```

### Response Format

```json
{
  "symbol": "AAPL",
  "market": "us",
  "model_type": "cnn",
  "window_size": 10,
  "image_size": 256,
  "predictions": {
    "T+1": {
      "predicted_class": "Up",
      "class_probabilities": {"Down": 0.35, "Up": 0.65},
      "predicted_return": 0.008
    },
    "T+3": {...},
    "T+5": {...}
  }
}
```

---

## Project Structure

```
k-line-analyze/
├── core/                          # Core functional modules
│   ├── config.py                 # Configuration management
│   ├── data_fetcher.py           # Stock data acquisition
│   ├── similarity_search.py      # FAISS similarity search
│   ├── feature_extractor.py      # Feature extraction
│   └── utils/                    # Utility functions
│       ├── kline_renderer.py     # K-line rendering
│       └── gaf_transformer.py    # GAF transformation
│
├── models/                        # Model definitions
│   ├── cnn_encoder.py            # CNN encoder
│   ├── cnn_predictor_v2.py       # CNN predictor v2
│   ├── transformer_predictor.py  # Transformer predictor
│   └── prediction_dataset_v2.py  # Prediction dataset
│
├── src/                           # Extension modules
│   ├── data/
│   │   ├── image_generator.py    # Image generator (OHLC/GAF/Hybrid)
│   │   └── dataset.py            # Advanced datasets
│   └── models/
│       ├── cnn_model.py          # CNN model variants
│       └── lightweight_cnn.py    # Lightweight CNN
│
├── backend/                       # FastAPI backend service
│   └── app/
│       ├── main.py               # API entry point
│       ├── api/routes/           # API routes
│       └── services/             # Business services
│
├── scripts/                       # Training and experiment scripts
│   ├── train_grouped_h20.py      # Grouped training (H20)
│   ├── train_cnn_predictor_v2.py # CNN v2 training
│   └── download_us_stocks.py     # Data download
│
├── data/                          # Data directory
│   └── raw/us/                   # US stock data
│
├── outputs/                       # Output directory
│   └── models/                   # Saved model weights
│
└── docs/                          # Documentation
```

---

## Core Technologies

### Image Encoding Methods

| Method | Description | Paper Reference |
|------|------|----------|
| **Candlestick** | Traditional candlestick charts | - |
| **OHLC Bars** | Sparse bar charts (recommended) | Xiu et al. (2021) |
| **GAF** | Gramian Angular Field | Chen & Tsai (2020) |
| **Hybrid** | Multi-channel fusion | - |

### Model Architectures

| Architecture | Input | Characteristics |
|------|------|------|
| **CNN (ResNet18)** | K-line images 256×256 | Image recognition, pretrained weights |
| **Grouped Models** | Trained by sector | Specialized optimization |

### Key Findings

1. **Grouped Training** is the most effective improvement method (+7%)
2. **OHLC Encoding** outperforms traditional candlestick charts
3. **20-day window** + **128px** is the optimal balance
4. **ResNet18** + **pretrained weights** is critical

---

## Documentation Navigation

| Document | Content |
|------|------|
| [THEORY.md](./THEORY.md) | Complete theoretical framework, data encoding methods, model architecture, training methodology |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | Complete experiment report: SAK-Net (57.81%), baseline comparison, grouped training, failure analysis |

---

## License

MIT License - See [LICENSE](LICENSE) file for details

> ⚠️ **Disclaimer**: This project is for academic research purposes only and does not constitute investment advice. The stock market involves risks; invest with caution.
