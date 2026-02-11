# 股票预测模型使用说明

## 概览

本项目在原有 K-Line 相似度搜索的基础上，新增了两类**股价走势预测模型**，各自支持「全体股票」和「单一股票」两种训练模式。目前聚焦于**美股 (US) 市场**。

| 模型 | 输入 | 核心思想 |
|------|------|----------|
| **CNN (k-line-net-mc)** | 10 天 K 线图 256×256 RGB | 图像识别蜡烛形态 |
| **Transformer** | 10 天 × 9 维数值序列 | 自注意力捕捉时序依赖 |

### 预测类别（二分类）

| 标签 | 值 | 条件 |
|------|---|------|
| **跌** | 0 | 未来收盘价 < 当前收盘价 |
| **涨** | 1 | 未来收盘价 >= 当前收盘价 |

准确率 = 预测正确的样本数 / 总样本数，直觉明了。

### 预测时间窗口

默认预测未来 **T+1、T+3、T+5** 天的涨跌（可通过 `--horizons` 自定义）。

---

## 文件结构

```
backend/app/models/
├── cnn_predictor_v2.py         # CNN 预测模型 (v2, 10天/256px, 二分类)
├── transformer_predictor.py    # Transformer 预测模型 (二分类)
├── prediction_dataset_v2.py    # 数据集 + 时间切分
├── cnn_predictor.py            # CNN v1 (旧版)
└── prediction_dataset.py       # 数据集 v1

scripts/
├── train_cnn_predictor_v2.py        # CNN v2 训练脚本 ← 推荐
├── train_transformer_predictor_v2.py # Transformer v2 训练脚本 ← 推荐
├── train_cnn_predictor.py            # CNN v1 训练脚本
└── train_transformer_predictor.py    # Transformer v1 训练脚本

backend/app/api/routes/
├── prediction_v2.py            # v2 预测 API
└── prediction.py               # v1 预测 API
```

---

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 训练模型

#### CNN v2 (推荐 — 图像识别路线)

```bash
# 全体美股训练（universal model）
python scripts/train_cnn_predictor_v2.py --mode all --epochs 50

# 单一股票训练（specialized model）
python scripts/train_cnn_predictor_v2.py --mode single --symbol AAPL --market us

# 单一股票 — NVDA
python scripts/train_cnn_predictor_v2.py --mode single --symbol NVDA --market us --epochs 80
```

#### Transformer v2

```bash
# 全体美股训练
python scripts/train_transformer_predictor_v2.py --mode all --epochs 80

# 单一股票训练
python scripts/train_transformer_predictor_v2.py --mode single --symbol NVDA --market us

# GPU 优化（H20 等高显存卡）
python scripts/train_transformer_predictor_v2.py --mode all \
    --batch-size 256 --d-model 256 --nhead 8 --layers 6 --amp
```

### 3. 模型输出位置

```
backend/trained_models/predictors/
├── cnn_v2_universal.pt                # CNN 全体模型
├── cnn_v2_us_AAPL.pt                 # CNN 单一股票模型
├── transformer_v2_universal.pt        # Transformer 全体模型
├── transformer_v2_us_NVDA.pt          # Transformer 单一股票模型
├── *_history.json                     # 训练损失/准确率曲线
└── *_config.json                      # 模型配置参数
```

---

## 训练参数详解

### CNN v2 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `all` | `all` = 全体股票, `single` = 单一股票 |
| `--symbol` | - | 股票代码（single 模式必填） |
| `--market` | `us` | 市场（目前聚焦 us） |
| `--window-size` | `10` | 每张图包含的 K 线天数 |
| `--image-size` | `256` | 图像分辨率 |
| `--horizons` | `1 3 5` | 预测时间窗口（天） |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `32` | 批量大小 |
| `--lr` | `1e-4` | 学习率 |
| `--freeze-backbone` | `5` | 冻结 ResNet 骨干的轮数 |
| `--val-ratio` | `0.15` | 验证集比例 |
| `--start-date` | 8 年前 | 数据起始日期 (YYYY-MM-DD) |
| `--end-date` | 今天 | 数据结束日期 (YYYY-MM-DD) |

### Transformer v2 额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--d-model` | `128` | Transformer 隐藏维度 |
| `--nhead` | `8` | 注意力头数 |
| `--layers` | `4` | Transformer 编码器层数 |
| `--dim-ff` | `512` | 前馈层维度 |
| `--dropout` | `0.1` | Dropout 率 |
| `--warmup-epochs` | `5` | 学习率预热轮数 |
| `--amp` | `false` | 启用混合精度训练 (FP16) |

---

## API 使用

启动后端：

```bash
cd backend
python -m uvicorn app.main:app --reload
```

### 端点一览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/prediction/v2/models` | GET | 列出所有已训练的 v2 模型 |
| `/api/v1/prediction/v2/predict` | POST | 使用指定模型进行预测 |
| `/api/v1/prediction/v2/compare/{symbol}` | GET | 对比所有 v2 模型的预测结果 |

### 预测请求示例

```bash
# Transformer 全体模型预测 AAPL
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "market": "us", "model_type": "transformer"}'

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
  "model_type": "transformer",
  "window_size": 10,
  "image_size": 256,
  "predictions": {
    "T+1": {
      "predicted_class": "涨",
      "class_probabilities": {
        "跌": 0.35,
        "涨": 0.65
      },
      "predicted_return": 0.008
    },
    "T+3": { "..." : "..." },
    "T+5": { "..." : "..." }
  }
}
```

---

## 设计说明

### 为什么二分类（涨/跌）而不是多分类？

- **准确率直觉**：50% 是随机基线，超过即有预测能力
- **类别均衡**：涨/跌天数大致各半，不需要复杂的类别权重调整
- **实用性**：投资决策的核心问题就是「买还是不买」

### 为什么 v2 用 10 天 + 256×256？

| 项目 | v1 (60天/128px) | v2 (10天/256px) |
|------|-----------------|-----------------|
| 每根蜡烛宽度 | ~2px (body 1px) | **~25px (body 19px)** |
| K 线形态可辨性 | 无法区分形态 | 十字星、锤子、吞没等清晰可辨 |
| ResNet 第一层 7×7 卷积覆盖 | ~3-4 天 | **~2-3 根蜡烛** (有意义的局部形态) |

### 为什么用时间切分验证集？

随机切分会导致相邻窗口分散到训练和验证集（98% 重叠），验证指标虚高。
时间切分确保验证集全部在训练集之后，模拟真实的「用历史预测未来」场景。

### 全体模型 vs 单一股票模型

| | 全体模型 (universal) | 单一股票模型 (single) |
|---|---|---|
| 训练数据 | 26 只美股 | 仅一只股票 |
| 泛化能力 | 强，可预测未见过的股票 | 弱，只适用于该股票 |
| 专精度 | 一般 | 可能更好捕捉个股特有模式 |
| 推荐用法 | 默认使用 | 对重点关注的个股额外训练 |

---

## 内建美股列表（26 只）

```
AAPL  MSFT  GOOGL  AMZN  NVDA  TSLA  META  AMD
NFLX  INTC  JPM    BAC   V     MA    JNJ   PFE
UNH   PG    KO     WMT   XOM   CVX   BA    DIS
CRM   CSCO
```

覆盖科技、金融、医疗、消费、能源五大板块。
