# V2 完整改进方案总结

## 所有改进已实施完成 ✅

### 1. 图像表示改进 ✅

#### OHLC 条形图 (`src/data/image_generator.py`)
- **方法**: `draw_ohlc_bars()`
- **特点**: 
  - 每交易日 3 像素宽度（开盘、高低、收盘）
  - 黑色背景 + 白色条形（稀疏表示）
  - 基于 Xiu et al. (2021) 论文

#### GAF 编码 (`src/data/image_generator.py`)
- **方法**: `create_gaf_ohlc()`
- **特点**:
  - Gramian Angular Summation Field (GASF)
  - 保留时间依赖性和相关性
  - 基于 Chen & Tsai (2020) 论文

#### 配置
- 窗口: 10 天 (vs 20 天)
- 分辨率: 256×256 (vs 128×128)
- 每根 K 线: ~25 像素宽 (vs ~6 像素)

---

### 2. 多尺度特征融合 ✅

#### 文件: `src/models/multiscale_cnn.py`

**三种架构**:

1. **MultiScaleKLineEncoder**
   - 独立 ResNet18 编码器处理 5/10/20 天
   - 跨尺度自注意力机制
   - 可学习的尺度权重

2. **HierarchicalMultiScaleCNN**
   - 渐进式融合（5d→10d→20d）
   - 层次化特征提取

3. **LightweightMultiScaleCNN**
   - 共享权重编码器
   - 多任务输出头
   - 软集成学习

#### 文件: `src/data/multiscale_dataset.py`

- 同时生成 5/10/20 天窗口图像
- 保持对齐的起始索引
- 支持所有图表类型

---

### 3. 高级标签策略 ✅

#### 文件: `src/data/advanced_labeling.py`

**四种策略**:

1. **Volatility (波动率调整)**
   ```python
   threshold = 0.5 * vol * sqrt(horizon)
   ```
   - 自适应阈值基于历史波动率
   - 过滤横盘样本

2. **Quantile (分位数)**
   - 基于历史收益分布
   - 顶部/底部分位数定义涨跌
   - 中间区域为横盘

3. **Regime (市场状态)**
   - 检测趋势/震荡/高波动/低波动
   - 不同状态使用不同阈值
   - 趋势市场需要更大波动

4. **Consensus (多时间尺度共识)**
   - 1/3/5/10 天预测的一致性投票
   - 高置信度信号

**工具类**:
- `SmartThresholdCalculator`: 预计算每支股票的阈值
- `filter_extreme_samples`: 过滤异常样本
- `balance_classes`: 类别平衡

---

### 4. 迁移学习 ✅

#### 文件: `scripts/train_transfer_learning.py`

**两阶段训练**:

1. **预训练阶段**
   ```bash
   python scripts/train_transfer_learning.py --stage pretrain --markets us cn
   ```
   - 使用所有市场数据训练通用模型
   - 学习通用 K 线模式
   - 高学习率 (1e-3)

2. **微调阶段**
   ```bash
   python scripts/train_transfer_learning.py --stage finetune --target-market us
   ```
   - 加载预训练权重
   - 冻结 backbone 前几轮
   - 低学习率 (1e-4)
   - 针对特定市场/股票优化

**关键参数**:
- `--freeze-epochs`: 冻结 backbone 的轮数
- `--lr-finetune`: 微调学习率 (通常是预训练的 1/10)

---

### 5. 集成学习 ✅

#### 文件: `scripts/train_ensemble.py`

**两种集成策略**:

1. **Diverse Ensemble**
   - 不同架构 (ResNet18, EfficientNet)
   - 不同图表类型 (OHLC, GAF, Hybrid)
   - 不同随机种子
   - 可学习的集成权重

2. **Multi-Scale Ensemble**
   - 融合三种多尺度模型
   - 注意力 + 层次化 + 轻量级

**集成方法**:
- 平均集成 (Average)
- 加权集成 (Weighted by validation accuracy)
- 投票集成 (Majority Voting)

**使用**:
```bash
python scripts/train_ensemble.py --type diverse --num-models 5
python scripts/train_ensemble.py --type multiscale
python scripts/train_ensemble.py --type both
```

---

## 快速开始

### 1. 基础实验 (OHLC + GAF)

```bash
python scripts/run_v2_experiments.py --experiment all --num-runs 3
```

### 2. 完整 V2 流程

```bash
# 运行所有阶段
python scripts/run_complete_v2.py --stage all

# 仅运行特定阶段
python scripts/run_complete_v2.py --stage baseline
python scripts/run_complete_v2.py --stage multiscale
```

### 3. 迁移学习

```bash
# 完整流程
python scripts/train_transfer_learning.py --stage both \
    --markets us cn \
    --target-market us \
    --epochs-pretrain 30 \
    --epochs-finetune 20
```

### 4. 集成模型

```bash
# 训练集成
python scripts/train_ensemble.py --type both --num-models 5
```

### 5. 可视化

```bash
# 比较不同表示方法
python scripts/visualize_representations.py --symbol AAPL
```

---

## 预期效果

| 方法 | 预期准确率 | 改进幅度 |
|------|-----------|---------|
| 原版 (CNN-Basic) | 51.8% | - |
| OHLC 10d/256 | 54-56% | +2-4% |
| GAF 编码 | 55-58% | +3-6% |
| 多尺度融合 | 56-60% | +4-8% |
| + 迁移学习 | 57-61% | +5-9% |
| + 集成模型 | 58-63% | +6-11% |

---

## 文件清单

### 新增文件

```
src/models/
├── multiscale_cnn.py          # 多尺度 CNN 模型 (3种架构)

src/data/
├── multiscale_dataset.py       # 多尺度数据集
├── advanced_labeling.py        # 高级标签策略

scripts/
├── run_v2_experiments.py       # V2 基础实验
├── run_complete_v2.py          # 完整 V2 流程
├── train_transfer_learning.py  # 迁移学习
├── train_ensemble.py           # 集成学习
├── visualize_representations.py # 可视化工具

V2_IMPROVEMENTS.md              # 初步改进文档
V2_COMPLETE_SUMMARY.md          # 本文档
```

### 修改文件

```
src/data/
├── image_generator.py          # + OHLC, GAF, Hybrid 方法
└── dataset.py                  # + chart_type, use_gaf 参数

src/models/
└── cnn_model.py                # 优化 ResNet18 支持 256x256

scripts/
└── run_baseline_experiments.py # + V2 实验配置
```

---

## 关键超参数建议

### 基础训练
- `window_size`: 10
- `img_size`: (256, 256)
- `batch_size`: 64 (256x256 图像更大)
- `lr`: 3e-4
- `epochs`: 30

### 多尺度训练
- `scales`: [5, 10, 20]
- `batch_size`: 32 (更大内存需求)
- `lr`: 2e-4

### 迁移学习
- 预训练 `lr`: 1e-3
- 微调 `lr`: 1e-4
- `freeze_epochs`: 5

### 集成学习
- `num_models`: 3-5
- 不同 `seed`, `chart_type`, `arch`

---

## 后续优化方向

1. **超参数搜索**: 使用 Optuna 自动调优
2. **更大数据集**: 加入更多国际市场数据
3. **注意力可视化**: 理解模型关注哪些 K 线形态
4. **在线学习**: 模型随时间自适应更新
5. **多任务学习**: 同时预测涨跌 + 波动率 + 趋势强度

---

## 参考文献

1. **Xiu et al. (2021)** - "(Re-)Imag(in)ing Price Trends"
   - OHLC 条形图, 多尺度, 迁移学习

2. **Chen & Tsai (2020)** - "Encoding candlesticks as images"
   - GAF 编码, 时间序列转图像

3. **Duong et al. (2025)** - "Investigating Market Strength Prediction"
   - 纯 CNN 优于模式检测
