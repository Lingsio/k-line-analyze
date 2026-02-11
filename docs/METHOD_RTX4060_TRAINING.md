# RTX 4060 分组训练指南

专为 NVIDIA RTX 4060 8GB VRAM 优化的分组训练方案。

## 特点

- **轻量级 CNN**: 仅 50万 参数（ResNet18 的 1/20）
- **显存优化**: Batch Size 32 仅需 ~4-6GB VRAM
- **产业分组**: 6大板块独立训练
- **3分类支持**: 涨/中性/跌

## 快速开始

### 1. 测试环境

```bash
python scripts/test_4060_setup.py
```

这会检测 GPU、测试模型显存占用、验证训练循环。

### 2. 训练所有板块

```bash
# 3分类模式（推荐）
python scripts/train_grouped_4060.py --all

# 2分类模式
python scripts/train_grouped_4060.py --all --num-classes 2

# 使用更小的模型
python scripts/train_grouped_4060.py --all --variant ultra
```

### 3. 训练单个板块

```bash
# 查看所有板块
python scripts/train_grouped_4060.py --sector Tech_Software

# 可选板块:
# - Tech_Semiconductors (半导体)
# - Tech_Software (软件与互联网)
# - Financials (金融)
# - Healthcare (医疗)
# - Consumer (消费品)
# - Industrials_Energy (工业与能源)
```

### 4. 调试模式

```bash
# 快速测试（每个板块只训练5个epoch）
python scripts/train_grouped_4060.py --all --debug
```

## 模型对比

| 模型 | 参数量 | 显存占用(batch=32) | 推荐场景 |
|------|--------|-------------------|---------|
| UltraLight | 10万 | ~2-3GB | 显存极紧张 |
| Light | 50万 | ~4-6GB | **推荐** |
| ResNet18 | 1100万 | ~6-8GB | 大显存GPU |

## 关键参数

### 数据参数
- `window_size`: 20天（回顾周期）
- `prediction_horizon`: 5天（预测周期）
- `label_threshold`: 1%（中性区间阈值）

### 训练参数
- `batch_size`: 32（4060 推荐）
- `epochs`: 50（早停 patience=10）
- `lr`: 0.001
- `dropout`: 0.3

## 防数据泄漏

关键设计：**只在训练集进行筛选**

```python
# 训练集：可以筛选
train_ds = StockDataset(
    mode='train',
    train_filter_threshold=0.003,  # 只用于训练
    ...
)

# 验证/测试集：不筛选（保持真实分布）
val_ds = StockDataset(mode='val', ...)   # 无筛选
test_ds = StockDataset(mode='test', ...) # 无筛选
```

## 输出结构

```
outputs/grouped_models_4060/
├── Tech_Software_3class.pt
├── Tech_Semiconductors_3class.pt
├── Financials_3class.pt
├── Healthcare_3class.pt
├── Consumer_3class.pt
├── Industrials_Energy_3class.pt
└── summary.json
```

## 性能预期

基于 RTX 4060 8GB：
- 每个板块训练时间: ~10-30 分钟（取决于数据量）
- 总训练时间: ~2-3 小时（6个板块）
- 预期准确率: 50-55%（金融预测具有挑战性）

## 故障排除

### 显存不足 (OOM)

```bash
# 减小 batch size
python scripts/train_grouped_4060.py --all --batch-size 16

# 使用 ultra 模型
python scripts/train_grouped_4060.py --all --variant ultra
```

### 训练太慢

```bash
# 增加 workers（根据你的CPU核心数）
# 编辑 scripts/train_grouped_4060.py:
# RTX4060_CONFIG['num_workers'] = 8
```

### 样本不足

某些板块可能股票较少，如果训练样本 < 50 会跳过。

## 进阶使用

### 自定义参数

编辑 `scripts/train_grouped_4060.py` 中的 `RTX4060_CONFIG`：

```python
RTX4060_CONFIG = {
    'batch_size': 32,
    'window_size': 20,          # 改为 10 或 30
    'prediction_horizon': 5,    # 改为 1（次日）或 10
    'label_threshold': 0.01,    # 改为 0.015（1.5%）
    'num_classes': 3,           # 2 或 3
    'epochs': 50,
    'lr': 1e-3,
    'dropout': 0.3,
}
```

### 加载训练好的模型

```python
import torch
from src.models.lightweight_cnn import build_lightweight_cnn

# 加载
ckpt = torch.load('outputs/grouped_models_4060/Tech_Software_3class.pt')
config = ckpt['config']

# 重建模型
model = build_lightweight_cnn(
    variant=config['cnn_variant'],
    num_classes=config['num_classes']
)
model.load_state_dict(ckpt['model_state_dict'])

# 测试指标
print(f"Test Accuracy: {ckpt['test_metrics']['accuracy']:.4f}")
print(f"Test F1: {ckpt['test_metrics']['f1']:.4f}")
```

## 下一步

1. 运行 `test_4060_setup.py` 验证环境
2. 运行 `train_grouped_4060.py --all --debug` 快速测试
3. 正式训练 `train_grouped_4060.py --all`
4. 查看结果 `outputs/grouped_models_4060/summary.json`
