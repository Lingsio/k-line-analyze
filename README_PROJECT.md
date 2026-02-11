# K-Line Visual Pattern Recognition System

> 基于深度学习的 K 线形态识别与股价预测系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 项目简介

本项目是一个**基于深度学习的 K 线形态相似性搜索与股价预测系统**。核心创新在于将股票 K 线数据转换为图像，利用 CNN 的图像识别能力自动学习价格模式，实现：

- **🔍 相似形态搜索**: 基于 FAISS 向量数据库快速匹配历史相似 K 线形态
- **📈 股价趋势预测**: CNN/Transformer 模型预测未来涨跌方向
- **🏭 分组专业模型**: 按行业板块训练专门模型，显著提升准确率

### 核心成果

| 指标 | 结果 |
|------|------|
| **平均准确率** | **57.81%** (目标 54%) |
| **最佳板块** | Consumer (60.37%) |
| **覆盖股票** | 57 只美股 |
| **行业板块** | 6 大板块 |

---

## 📖 理论框架

### 核心思想

> "K 线图的视觉模式包含可预测未来价格走势的有效信息"

传统技术分析依赖人工识别 K 线形态（十字星、锤子线、吞没等），存在主观性强、效率低下等问题。本项目将 K 线图视为图像，利用 CNN 自动学习模式：

```
OHLCV 数据 → 图像编码 → CNN 特征提取 → 涨跌预测
```

### 编码方式

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
| **Transformer** | 序列 10×9 | 自注意力，时序建模 |
| **分组模型** | 按板块训练 | 专业优化，异质处理 |

---

## 🚀 快速开始

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

## 📊 实验成果

### SAK-Net 实验结果 (H20, 2026-02-11)

| 板块 | 股票数 | 准确率 | F1 | 状态 |
|------|--------|--------|-----|------|
| **Consumer** 🏆 | 10 | **60.37%** | 0.59 | ✅ 推荐部署 |
| **Industrials_Energy** | 8 | **59.94%** | 0.52 | ✅ 推荐部署 |
| **Tech_Semiconductors** | 6 | **59.88%** | 0.57 | ✅ 推荐部署 |
| **Tech_Software** | 14 | **57.22%** | 0.58 | ✅ 可部署 |
| **Financials** | 9 | **55.90%** | 0.55 | ⚠️ 需优化 |
| **Healthcare** | 10 | **53.54%** | 0.52 | ⚠️ 需优化 |
| **平均** | **57** | **57.81%** | - | ✅ 超目标 |

### 股票覆盖

**6 大板块，57 只美股：**

| 板块 | 代表股票 |
|------|----------|
| Tech_Semiconductors | NVDA, AMD, INTC, QCOM, TXN, AVGO |
| Tech_Software | MSFT, AAPL, GOOGL, META, AMZN, NFLX, TSLA... |
| Financials | JPM, GS, BLK, V, MA... |
| Healthcare | LLY, UNH, JNJ, PFE, ABBV... |
| Consumer | WMT, COST, MCD, DIS, NKE... |
| Industrials_Energy | BA, CAT, CVX, XOM, CSCO... |

---

## 🏗️ 项目结构

```
k-line-analyze/
├── core/                          # 核心功能模块
│   ├── config.py                 # 配置管理
│   ├── data_fetcher.py           # 股票数据获取
│   ├── similarity_search.py      # FAISS 相似性搜索
│   └── utils/                    # 工具函数
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
│   ├── train_transformer_predictor_v2.py # Transformer 训练
│   └── run_v2_experiments.py     # V2 实验运行
│
├── data/                          # 数据目录
│   └── raw/us/                   # 美股数据 (CSV/Parquet)
│
├── outputs/                       # 输出目录
│   ├── models/                   # 保存的模型权重
│   └── grouped_h20/              # 分组训练结果
│
└── docs/                          # 文档
    ├── THEORY.md                 # 理论框架详解
    ├── RESULTS_SAK_NET_2026-02-11.md  # SAK-Net 实验记录
    └── prediction-models.md      # 模型使用说明
```

---

## 🔬 技术亮点

### 1. 图像编码创新

**OHLC Bars (Xiu et al. 2021)**
```python
# 稀疏表示，每根 K 线 3 像素
- 开盘: 水平线段
- 高低: 垂直线段
- 收盘: 水平线段
```

**GAF 编码 (Chen & Tsai 2020)**
```python
# 保留时序相关性
1. 归一化到 [-1, 1]
2. 极坐标编码: φ = arccos(x̃)
3. GASF: cos(φᵢ + φⱼ)
```

### 2. 分组训练策略

```python
# 板块异质性假设
Tech_Semiconductors ≠ Healthcare  (不同模式)

# 独立训练
Sector_Model = ResNet18(tech_stocks)  # 学习技术周期模式
Sector_Model = ResNet18(health_stocks)  # 学习药物试验事件
```

**效果**: 从 51% → 58% (+7%)

### 3. 数据泄漏防护

```python
# ❌ 错误: 全数据集筛选
train_ds = Dataset(quantile_filter=0.35)
val_ds = Dataset(quantile_filter=0.35)    # 泄漏!
test_ds = Dataset(quantile_filter=0.35)   # 泄漏!

# ✅ 正确: 仅训练集筛选
train_ds = Dataset(quantile_filter=0.35)  # 仅训练
val_ds = Dataset()                        # 完整分布
test_ds = Dataset()                       # 完整分布
```

---

## 📈 性能对比

### 历史实验演进

| 阶段 | 方法 | 准确率 | 提升 |
|------|------|--------|------|
| Baseline | LSTM | 49.68% | - |
| Baseline | ResNet1D | 49.68% | 0% |
| CNN v1 | CNN (64×64) | 50.36% | +0.68% |
| CNN v2 | CNN (128×128) | 50.87% | +0.51% |
| + CLAHE | KLineNet | 50.83% | -0.04% |
| + MultiChannel | KLineNet-MC | 50.95% | +0.12% |
| **Grouped** | **分组训练** | **57.81%** | **+6.86%** 🚀 |

### 关键发现

1. **分组训练** 是最有效的提升手段 (+7%)
2. **OHLC 编码** 优于传统蜡烛图
3. **20天窗口** + **128px** 是最佳平衡
4. **ResNet18** + **预训练权重** 是关键

---

## 🛠️ 高级配置

### H20 (96GB VRAM) 配置

```python
H20_CONFIG = {
    'batch_size': 128,
    'num_workers': 8,
    'epochs': 50,
    'lr': 4e-4,
    'use_amp': True,        # 混合精度
    'use_compile': True,    # torch.compile
}
```

### RTX 4060 (8GB VRAM) 配置

```python
RTX4060_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'epochs': 50,
    'lr': 1e-3,
    'model': 'lightweight_cnn',  # 50万参数
}
```

---

## 📚 文档导航

| 文档 | 内容 |
|------|------|
| [THEORY.md](THEORY.md) | 完整理论框架、数学原理、技术实现 |
| [METHOD_IMAGE_ENCODING.md](METHOD_IMAGE_ENCODING.md) | 图像编码方法 (OHLC/GAF/Hybrid) |
| [METHOD_MULTISCALE_FUSION.md](METHOD_MULTISCALE_FUSION.md) | 多尺度 CNN 融合方法 |
| [METHOD_OPTIMIZED_PIPELINE.md](METHOD_OPTIMIZED_PIPELINE.md) | 6 阶段优化流水线 |
| [docs/RESULTS_SAK_NET_2026-02-11.md](docs/RESULTS_SAK_NET_2026-02-11.md) | **最新实验记录 (57.81%)** |
| [docs/USER_GUIDE_PREDICTION_MODELS.md](docs/USER_GUIDE_PREDICTION_MODELS.md) | 模型使用说明、API 文档 |

---

## 🤝 贡献指南

### 开发流程

```bash
# 1. Fork 仓库
# 2. 创建分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "feat: add your feature"

# 4. 推送分支
git push origin feature/your-feature

# 5. 创建 Pull Request
```

### 代码规范

- **命名**: 小写+下划线 (文件), PascalCase (类), snake_case (函数)
- **类型注解**: 重要函数必须添加
- **文档**: Google 风格 docstring

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- **Xiu et al. (2021)** - OHLC 条形图设计
- **Chen & Tsai (2020)** - GAF 编码方法
- **PyTorch Team** - 深度学习框架
- **FAISS Team** - 向量搜索引擎

---

## 📞 联系方式

- 项目主页: https://github.com/your-org/k-line-analyze
- 问题反馈: https://github.com/your-org/k-line-analyze/issues
- 邮件: your-email@example.com

---

> ⚠️ **免责声明**: 本项目仅供学术研究使用，不构成投资建议。股市有风险，投资需谨慎。
