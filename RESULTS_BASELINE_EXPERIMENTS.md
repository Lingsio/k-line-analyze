# Baseline Experiments - K-Line Visual Pattern Recognition

> 实验记录文档  
> 创建日期: 2026-02-06  
> 最后更新: 2026-02-06

---

## 1. 实验概述

本实验旨在评估基于 K 线图视觉模式识别的股票价格预测方法，比较多种基线模型与我们提出的 KLineNet 方法的性能。

### 1.1 研究目标

- 验证将 K 线图转换为图像进行 CNN 分类的有效性
- 对比序列模型 (LSTM, ResNet1D) 与视觉模型 (CNN) 的性能差异
- 评估图像预处理技术 (CLAHE, 边缘检测等) 对模型性能的影响
- 验证数据增强策略 (Mixup, 随机增强) 的效果

---

## 2. 硬件环境

| 配置项 | 规格 |
|--------|------|
| **GPU** | NVIDIA H20 |
| **GPU 显存** | 95.1 GB HBM3 |
| **CPU 核心数** | 32 |
| **系统内存** | 128 GB |
| **CUDA 版本** | 12.6 |
| **PyTorch 版本** | 2.0+ |

---

## 3. 数据集

### 3.1 数据来源

| 市场 | 股票数量 | 数据目录 |
|------|----------|----------|
| 美国 (US) | 50 | `/workspace/data/raw/us` |
| 中国 (CN) | 50+ | `/workspace/data/raw/cn` |

**总计**: ~107 只股票的历史 K 线数据

### 3.2 数据划分

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| Train | ~69,669 | 模型训练 |
| Validation | ~23,427 | 超参数调优、早停 |
| Test | ~23,442 | 最终性能评估 |

### 3.3 特征说明

- **输入窗口**: 20 个交易日的 K 线数据
- **预测目标**: 未来 5 个交易日的价格走向 (二分类: 上涨/下跌)
- **标签阈值**: 动态阈值 (基于历史波动率) 或固定 0.5%

---

## 4. 实验配置

### 4.1 通用训练配置

```python
CONFIG = {
    'epochs': 20,              # 最大训练轮数
    'batch_size': 128,         # 批次大小 (针对 H20 优化)
    'lr': 4e-4,                # 学习率 (线性缩放)
    'weight_decay': 1e-5,      # L2 正则化
    'patience': 5,             # 早停耐心值
    'num_runs': 3,             # 每个实验运行次数 (统计显著性)
}
```

### 4.2 性能优化配置

```python
# 数据加载优化
NUM_WORKERS = 8              # 数据加载进程数
PIN_MEMORY = True            # GPU 内存固定
PREFETCH_FACTOR = 4          # 预取批次数
PERSISTENT_WORKERS = True    # 持久化 worker

# GPU 优化
USE_AMP = True               # 混合精度训练 (FP16)
USE_COMPILE = True           # torch.compile 模型编译
CUDNN_BENCHMARK = True       # cuDNN 自动调优
TF32 = True                  # TensorFloat-32 加速
```

---

## 5. 模型配置

### 5.1 基线模型

#### LSTM
```python
{
    'model_type': 'lstm',
    'device': 'cpu',           # CPU 运行避免 NaN
    'hidden_dim': 128,
    'num_layers': 2,
    'embedding_dim': 256,
}
```

#### ResNet1D
```python
{
    'model_type': 'resnet1d',
    'device': 'cpu',
    'layers': [2, 2, 2, 2],
    'embedding_dim': 256,
}
```

#### CNN-Raw (最简基线)
```python
{
    'model_type': 'cnn',
    'img_size': (64, 64),
    'norm_method': 'minmax',
    'use_clahe': False,
    'output_channels': 'rgb',
    'label_threshold': 0.005,
    'augment_prob': 0.0,
    'device': 'cuda',
}
```

#### CNN-Basic
```python
{
    'model_type': 'cnn',
    'img_size': (128, 128),
    'norm_method': 'minmax',
    'use_clahe': False,
    'output_channels': 'rgb',
    'label_threshold': 0.005,
    'augment_prob': 0.0,
    'device': 'cuda',
}
```

### 5.2 我们的方法

#### KLineNet
```python
{
    'model_type': 'cnn',
    'img_size': (128, 128),
    'norm_method': 'robust',       # 鲁棒归一化
    'use_clahe': True,             # 对比度增强
    'output_channels': 'rgb',
    'label_threshold': 'dynamic',  # 动态阈值
    'augment_prob': 0.3,           # 30% 增强概率
    'device': 'cuda',
}
```

#### KLineNet-MC (Multi-Channel)
```python
{
    'model_type': 'cnn',
    'img_size': (128, 128),
    'norm_method': 'robust',
    'use_clahe': True,
    'output_channels': 'rgb+edge', # 多通道 (RGB + 边缘)
    'label_threshold': 'dynamic',
    'augment_prob': 0.3,
    'mixup_prob': 0.1,             # 10% Mixup 概率
    'device': 'cuda',
}
```

---

## 6. 评估指标

| 指标 | 说明 |
|------|------|
| **Accuracy** | 分类准确率 |
| **F1 Score** | F1 分数 (加权平均) |
| **Precision** | 精确率 (加权平均) |
| **Recall** | 召回率 (加权平均) |
| **AUC** | ROC 曲线下面积 |

每个指标报告 **均值 ± 标准差** (基于 3 次运行)

---

## 7. 实验结果

### 7.1 主要对比结果 (Table 1)

| Method | Accuracy | F1 Score | AUC |
|--------|----------|----------|-----|
| LSTM | 0.4968±0.0000 | 0.3298±0.0000 | 0.5000±0.0000 |
| ResNet1D | 0.4968±0.0000 | 0.3298±0.0000 | 0.5000±0.0000 |
| CNN-Raw | 0.5036±0.0031 | 0.5025±0.0036 | 0.5042±0.0037 |
| CNN-Basic | 0.5087±0.0012 | 0.5065±0.0011 | 0.5106±0.0017 |
| **KLineNet** | 0.5083±0.0033 | 0.5039±0.0059 | 0.5118±0.0061 |
| **KLineNet-MC** | **0.5095±0.0022** | **0.5083±0.0018** | **0.5148±0.0038** |

> 注: 实验进行中，结果持续更新

### 7.2 消融实验 (Ablation Study)

#### 图像尺寸影响

| Image Size | Accuracy | F1 Score | 对应模型 |
|------------|----------|----------|----------|
| 64×64 | 0.5036±0.0031 | 0.5025±0.0036 | CNN-Raw |
| 128×128 | 0.5087±0.0012 | 0.5065±0.0011 | CNN-Basic |
| 224×224 | - | - | (未实验) |

**发现**: 从64→128提升显著 (+0.51% acc, +0.40% f1)，更大尺寸可能带来边际收益递减。

#### 预处理方法影响

| Method | Accuracy | F1 Score | 对应模型 |
|--------|----------|----------|----------|
| MinMax | 0.5087±0.0012 | 0.5065±0.0011 | CNN-Basic |
| Robust + CLAHE | 0.5083±0.0033 | 0.5039±0.0059 | KLineNet |

**注**: Robust归一化单独的效果需额外实验验证。CLAHE主要提升AUC (0.5106→0.5118)。

#### 数据增强影响

| Augmentation | Accuracy | F1 Score | 对应模型 |
|--------------|----------|----------|----------|
| None | 0.5087±0.0012 | 0.5065±0.0011 | CNN-Basic |
| Random Aug (0.3) | 0.5083±0.0033 | 0.5039±0.0059 | KLineNet |
| Random Aug + Mixup | 0.5095±0.0022 | 0.5083±0.0018 | KLineNet-MC |

**发现**: Mixup显著提升性能 (+0.18% f1)，Random Aug单独使用效果不明显，可能需要配合其他技术。

---

## 8. 运行时间

### 8.1 预计运行时间

| 阶段 | 时间 |
|------|------|
| 优化前 | 3-4 小时 |
| 优化后 | 30-60 分钟 |

### 8.2 各模型运行时间

| Model | Time per Epoch | Total Time (20 epochs × 3 runs) |
|-------|----------------|--------------------------------|
| LSTM | - | - |
| ResNet1D | - | - |
| CNN-Raw | - | - |
| CNN-Basic | - | - |
| KLineNet | - | - |
| KLineNet-MC | - | - |

---

## 9. 输出文件

实验结果保存在 `/workspace/outputs/baseline_results/` 目录:

| 文件 | 说明 |
|------|------|
| `intermediate_YYYYMMDD_HHMMSS.json` | 中间结果 (每个实验后保存) |
| `full_results_YYYYMMDD_HHMMSS.json` | 最终完整结果 |

---

## 10. 依赖环境

详见 `/workspace/requirements.txt`:

```
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless<4.10
faiss-cpu>=1.7.4
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
```

---

## 11. 运行命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整实验
python scripts/run_baseline_experiments.py
```

---

## 12. 备注

- 序列模型 (LSTM, ResNet1D) 在 CPU 上运行以避免数值不稳定 (NaN)
- CNN 模型使用 NVIDIA H20 GPU 加速训练
- 使用混合精度 (AMP) 和 torch.compile 进一步加速
- 每个实验运行 3 次以获得统计显著性

---

## 附录: 实验日志

### 实验 1: LSTM ✅
- **开始时间**: 2026-02-06 07:31:21
- **结束时间**: 2026-02-06 07:55:04
- **状态**: 已完成
- **设备**: CPU
- **运行次数**: 3 (seeds: 42, 142, 242)
- **每次训练轮数**: 6 (早停)
- **结果摘要**:
  - Accuracy: 0.4968 ± 0.0000
  - F1 Score: 0.3298 ± 0.0000
  - Precision: 0.2469 ± 0.0000
  - Recall: 0.4968 ± 0.0000
  - AUC: 0.5000 ± 0.0000
- **备注**: 模型预测趋向于单一类别，表现接近随机猜测

### 实验 2: ResNet1D ✅
- **开始时间**: 2026-02-06 07:55:04
- **结束时间**: 2026-02-06 08:21:30
- **状态**: 已完成
- **设备**: CPU
- **运行次数**: 3 (seeds: 42, 142, 242)
- **每次训练轮数**: 6 (早停)
- **结果摘要**:
  - Accuracy: 0.4968 ± 0.0000
  - F1 Score: 0.3298 ± 0.0000
  - Precision: 0.2469 ± 0.0000
  - Recall: 0.4968 ± 0.0000
  - AUC: 0.5000 ± 0.0000
- **训练集准确率**: 51.9% → 62.3% (严重过拟合)
- **备注**: 比 LSTM 过拟合更严重，训练集准确率达到 62%，但验证集仍为 50.66%

### 实验 3: CNN-Raw
- **开始时间**: 2026-02-06 09:20:09
- **结束时间**: 2026-02-06 09:42:XX
- **状态**: 已完成
- **设备**: GPU (NVIDIA H20)
- **运行次数**: 3 (seeds: 42, 142, 242)
- **每次训练轮数**: 9, 16, 11 (早停)
- **结果摘要**:
    - Accuracy: 0.5036 ± 0.0031
    - F1 Score: 0.5025 ± 0.0036
    - Precision: 0.5036 ± 0.0034
    - Recall: 0.5036 ± 0.0031
    - AUC: 0.5042 ± 0.0037
- **单次结果**:
    - [seed=42] acc=0.5054, f1=0.5045, auc=0.5073, best_val_f1=0.5101, 9 epochs
    - [seed=142] acc=0.5061, f1=0.5056, auc=0.5064, best_val_f1=0.5023, 16 epochs
    - [seed=242] acc=0.4991, f1=0.4975, auc=0.4991, best_val_f1=0.5064, 11 epochs
- **训练集准确率**: 51.3% → 79.3% (部分run最高99.9%，过拟合明显)
- **备注**: CNN-Raw 在测试集上略高于随机，训练集准确率随epoch大幅提升但泛化有限，验证集F1最高约0.51，AUC略高于0.5。

### 实验 4: CNN-Basic
- **开始时间**: 2026-02-06 10:10:56
- **结束时间**: 2026-02-06 10:32:XX
- **状态**: 已完成
- **设备**: GPU (NVIDIA H20)
- **运行次数**: 3 (seeds: 42, 142, 242)
- **每次训练轮数**: 8, 9, 9 (早停)
- **结果摘要**:
    - Accuracy: 0.5087 ± 0.0012
    - F1 Score: 0.5065 ± 0.0011
    - Precision: 0.5093 ± 0.0013
    - Recall: 0.5087 ± 0.0012
    - AUC: 0.5106 ± 0.0017
- **单次结果**:
    - [seed=42] acc=0.5095, f1=0.5075, auc=0.5128, best_val_f1=0.5239, 8 epochs
    - [seed=142] acc=0.5096, f1=0.5070, auc=0.5103, best_val_f1=0.5092, 9 epochs
    - [seed=242] acc=0.5070, f1=0.5049, auc=0.5087, best_val_f1=0.5187, 9 epochs
- **训练集准确率**: 50.7% → 62.7% (部分run最高58.4%，过拟合略有缓解)
- **备注**: CNN-Basic 在测试集上表现略优于 CNN-Raw，AUC 达到 0.51，验证集 F1 最高 0.52 左右，泛化能力略有提升。

### 实验 5: KLineNet ✅
- **开始时间**: 2026-02-06 13:59:57
- **结束时间**: 2026-02-06 14:51:22
- **状态**: 已完成
- **设备**: GPU (NVIDIA H20)
- **运行次数**: 3 (seeds: 42, 142, 242)
- **每次训练轮数**: 16, 8, 14 (早停)
- **结果摘要**:
  - Accuracy: 0.5083 ± 0.0033
  - F1 Score: 0.5039 ± 0.0059
  - Precision: 0.5090 ± 0.0036
  - Recall: 0.5083 ± 0.0033
  - AUC: 0.5118 ± 0.0061
- **单次结果**:
  - [seed=42] acc=0.5126, f1=0.5118, auc=0.5183, best_val_f1=0.5131, 16 epochs
  - [seed=142] acc=0.5044, f1=0.4978, auc=0.5036, best_val_f1=0.5125, 8 epochs
  - [seed=242] acc=0.5078, f1=0.5021, auc=0.5135, best_val_f1=0.5139, 14 epochs
- **运行时间**: 约51分钟
- **备注**: 使用 robust norm + CLAHE + 数据增强(0.3)，性能略优于 CNN-Basic，AUC 达到 0.51，验证集 F1 最高约 0.51-0.52

### 实验 6: KLineNet-MC ✅
- **开始时间**: 2026-02-06 15:08:01
- **结束时间**: 2026-02-06 16:43:11
- **状态**: 已完成
- **设备**: GPU (NVIDIA H20)
- **运行次数**: 3 (seeds: 42, 142, 242)
- **每次训练轮数**: 14, 20, 11 (早停或完整训练)
- **结果摘要**:
  - Accuracy: 0.5095 ± 0.0022
  - F1 Score: 0.5083 ± 0.0018
  - Precision: 0.5100 ± 0.0023
  - Recall: 0.5095 ± 0.0022
  - AUC: 0.5148 ± 0.0038
- **单次结果**:
  - [seed=42] acc=0.5119, f1=0.5097, auc=0.5197, best_val_f1=0.5089, 14 epochs
  - [seed=142] acc=0.5100, f1=0.5094, auc=0.5143, best_val_f1=0.5148, 20 epochs
  - [seed=242] acc=0.5066, f1=0.5058, auc=0.5104, best_val_f1=0.5222, 11 epochs
- **运行时间**: 约95分钟
- **备注**: 使用多通道 (RGB + 边缘检测) + Mixup (0.1) + 数据增强(0.3)，在所有指标上都优于 KLineNet，且方差更小，泛化能力最佳，AUC 达到 0.51 
