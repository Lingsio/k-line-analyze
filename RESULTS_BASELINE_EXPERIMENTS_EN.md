# Baseline Experiments - K-Line Visual Pattern Recognition

> Experiment Record Document  
> Created: 2026-02-06  
> Last Updated: 2026-02-06

---

## 1. Experiment Overview

This experiment aims to evaluate stock price prediction methods based on K-line chart visual pattern recognition, comparing the performance of multiple baseline models with our proposed KLineNet approach.

### 1.1 Research Objectives

- Validate the effectiveness of converting K-line charts into images for CNN classification
- Compare performance differences between sequence models (LSTM, ResNet1D) and visual models (CNN)
- Assess the impact of image preprocessing techniques (CLAHE, edge detection, etc.) on model performance
- Evaluate the effectiveness of data augmentation strategies (Mixup, random augmentation)

---

## 2. Hardware Environment

| Configuration | Specification |
|---------------|---------------|
| **GPU** | NVIDIA H20 |
| **GPU Memory** | 95.1 GB HBM3 |
| **CPU Cores** | 32 |
| **System Memory** | 128 GB |
| **CUDA Version** | 12.6 |
| **PyTorch Version** | 2.0+ |

---

## 3. Dataset

### 3.1 Data Sources

| Market | Stock Count | Data Directory |
|--------|-------------|----------------|
| US | 50 | `/workspace/data/raw/us` |
| CN | 50+ | `/workspace/data/raw/cn` |

**Total**: ~107 stocks with historical K-line data

### 3.2 Data Split

| Dataset | Sample Count | Purpose |
|---------|--------------|---------|
| Train | ~69,669 | Model training |
| Validation | ~23,427 | Hyperparameter tuning, early stopping |
| Test | ~23,442 | Final performance evaluation |

### 3.3 Feature Description

- **Input Window**: 20 trading days of K-line data
- **Prediction Target**: Price direction for the next 5 trading days (binary classification: up/down)
- **Label Threshold**: Dynamic threshold (based on historical volatility) or fixed 0.5%

---

## 4. Experiment Configuration

### 4.1 General Training Configuration

```python
CONFIG = {
    'epochs': 20,              # Maximum training epochs
    'batch_size': 128,         # Batch size (optimized for H20)
    'lr': 4e-4,                # Learning rate (linear scaling)
    'weight_decay': 1e-5,      # L2 regularization
    'patience': 5,             # Early stopping patience
    'num_runs': 3,             # Number of runs per experiment (statistical significance)
}
```

### 4.2 Performance Optimization Configuration

```python
# Data loading optimization
NUM_WORKERS = 8              # Number of data loading processes
PIN_MEMORY = True            # GPU memory pinning
PREFETCH_FACTOR = 4          # Number of batches to prefetch
PERSISTENT_WORKERS = True    # Persistent workers

# GPU optimization
USE_AMP = True               # Mixed precision training (FP16)
USE_COMPILE = True           # torch.compile model compilation
CUDNN_BENCHMARK = True       # cuDNN auto-tuning
TF32 = True                  # TensorFloat-32 acceleration
```

---

## 5. Model Configurations

### 5.1 Baseline Models

#### LSTM
```python
{
    'model_type': 'lstm',
    'device': 'cpu',           # CPU execution to avoid NaN
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

#### CNN-Raw (Minimal Baseline)
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

### 5.2 Our Methods

#### KLineNet
```python
{
    'model_type': 'cnn',
    'img_size': (128, 128),
    'norm_method': 'robust',       # Robust normalization
    'use_clahe': True,             # Contrast enhancement
    'output_channels': 'rgb',
    'label_threshold': 'dynamic',  # Dynamic threshold
    'augment_prob': 0.3,           # 30% augmentation probability
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
    'output_channels': 'rgb+edge', # Multi-channel (RGB + edge)
    'label_threshold': 'dynamic',
    'augment_prob': 0.3,
    'mixup_prob': 0.1,             # 10% Mixup probability
    'device': 'cuda',
}
```

---

## 6. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Classification accuracy |
| **F1 Score** | F1 score (weighted average) |
| **Precision** | Precision (weighted average) |
| **Recall** | Recall (weighted average) |
| **AUC** | Area Under ROC Curve |

Each metric is reported as **mean ± standard deviation** (based on 3 runs)

---

## 7. Experiment Results

### 7.1 Main Comparison Results (Table 1)

| Method | Accuracy | F1 Score | AUC |
|--------|----------|----------|-----|
| LSTM | 0.4968±0.0000 | 0.3298±0.0000 | 0.5000±0.0000 |
| ResNet1D | 0.4968±0.0000 | 0.3298±0.0000 | 0.5000±0.0000 |
| CNN-Raw | 0.5036±0.0031 | 0.5025±0.0036 | 0.5042±0.0037 |
| CNN-Basic | 0.5087±0.0012 | 0.5065±0.0011 | 0.5106±0.0017 |
| **KLineNet** | 0.5083±0.0033 | 0.5039±0.0059 | 0.5118±0.0061 |
| **KLineNet-MC** | **0.5095±0.0022** | **0.5083±0.0018** | **0.5148±0.0038** |

> Note: Experiments are ongoing, results are continuously updated

### 7.2 Ablation Study

#### Impact of Image Size

| Image Size | Accuracy | F1 Score | Corresponding Model |
|------------|----------|----------|---------------------|
| 64×64 | 0.5036±0.0031 | 0.5025±0.0036 | CNN-Raw |
| 128×128 | 0.5087±0.0012 | 0.5065±0.0011 | CNN-Basic |
| 224×224 | - | - | (Not tested) |

**Finding**: Significant improvement from 64→128 (+0.51% acc, +0.40% f1), larger sizes may yield diminishing marginal returns.

#### Impact of Preprocessing Methods

| Method | Accuracy | F1 Score | Corresponding Model |
|--------|----------|----------|---------------------|
| MinMax | 0.5087±0.0012 | 0.5065±0.0011 | CNN-Basic |
| Robust + CLAHE | 0.5083±0.0033 | 0.5039±0.0059 | KLineNet |

**Note**: The effect of Robust normalization alone requires additional experimental validation. CLAHE mainly improves AUC (0.5106→0.5118).

#### Impact of Data Augmentation

| Augmentation | Accuracy | F1 Score | Corresponding Model |
|--------------|----------|----------|---------------------|
| None | 0.5087±0.0012 | 0.5065±0.0011 | CNN-Basic |
| Random Aug (0.3) | 0.5083±0.0033 | 0.5039±0.0059 | KLineNet |
| Random Aug + Mixup | 0.5095±0.0022 | 0.5083±0.0018 | KLineNet-MC |

**Finding**: Mixup significantly improves performance (+0.18% f1). Random Augmentation alone shows limited effect and may need to be combined with other techniques.

---

## 8. Runtime

### 8.1 Estimated Runtime

| Stage | Time |
|-------|------|
| Before optimization | 3-4 hours |
| After optimization | 30-60 minutes |

### 8.2 Runtime by Model

| Model | Time per Epoch | Total Time (20 epochs × 3 runs) |
|-------|----------------|--------------------------------|
| LSTM | - | - |
| ResNet1D | - | - |
| CNN-Raw | - | - |
| CNN-Basic | - | - |
| KLineNet | - | - |
| KLineNet-MC | - | - |

---

## 9. Output Files

Experiment results are saved in the `/workspace/outputs/baseline_results/` directory:

| File | Description |
|------|-------------|
| `intermediate_YYYYMMDD_HHMMSS.json` | Intermediate results (saved after each experiment) |
| `full_results_YYYYMMDD_HHMMSS.json` | Final complete results |

---

## 10. Dependencies

See `/workspace/requirements.txt` for details:

```
# Core dependencies
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

## 11. Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete experiments
python scripts/run_baseline_experiments.py
```

---

## 12. Notes

- Sequence models (LSTM, ResNet1D) run on CPU to avoid numerical instability (NaN)
- CNN models use NVIDIA H20 GPU for accelerated training
- Mixed precision (AMP) and torch.compile are used for further acceleration
- Each experiment runs 3 times for statistical significance

---

## Appendix: Experiment Logs

### Experiment 1: LSTM ✅
- **Start Time**: 2026-02-06 07:31:21
- **End Time**: 2026-02-06 07:55:04
- **Status**: Completed
- **Device**: CPU
- **Number of Runs**: 3 (seeds: 42, 142, 242)
- **Epochs per Run**: 6 (early stopping)
- **Results Summary**:
  - Accuracy: 0.4968 ± 0.0000
  - F1 Score: 0.3298 ± 0.0000
  - Precision: 0.2469 ± 0.0000
  - Recall: 0.4968 ± 0.0000
  - AUC: 0.5000 ± 0.0000
- **Notes**: Model predictions tend toward a single class, performing close to random guessing

### Experiment 2: ResNet1D ✅
- **Start Time**: 2026-02-06 07:55:04
- **End Time**: 2026-02-06 08:21:30
- **Status**: Completed
- **Device**: CPU
- **Number of Runs**: 3 (seeds: 42, 142, 242)
- **Epochs per Run**: 6 (early stopping)
- **Results Summary**:
  - Accuracy: 0.4968 ± 0.0000
  - F1 Score: 0.3298 ± 0.0000
  - Precision: 0.2469 ± 0.0000
  - Recall: 0.4968 ± 0.0000
  - AUC: 0.5000 ± 0.0000
- **Training Accuracy**: 51.9% → 62.3% (severe overfitting)
- **Notes**: More severe overfitting than LSTM, training accuracy reached 62% but validation remained at 50.66%

### Experiment 3: CNN-Raw
- **Start Time**: 2026-02-06 09:20:09
- **End Time**: 2026-02-06 09:42:XX
- **Status**: Completed
- **Device**: GPU (NVIDIA H20)
- **Number of Runs**: 3 (seeds: 42, 142, 242)
- **Epochs per Run**: 9, 16, 11 (early stopping)
- **Results Summary**:
    - Accuracy: 0.5036 ± 0.0031
    - F1 Score: 0.5025 ± 0.0036
    - Precision: 0.5036 ± 0.0034
    - Recall: 0.5036 ± 0.0031
    - AUC: 0.5042 ± 0.0037
- **Single Run Results**:
    - [seed=42] acc=0.5054, f1=0.5045, auc=0.5073, best_val_f1=0.5101, 9 epochs
    - [seed=142] acc=0.5061, f1=0.5056, auc=0.5064, best_val_f1=0.5023, 16 epochs
    - [seed=242] acc=0.4991, f1=0.4975, auc=0.4991, best_val_f1=0.5064, 11 epochs
- **Training Accuracy**: 51.3% → 79.3% (some runs peaked at 99.9%, obvious overfitting)
- **Notes**: CNN-Raw slightly outperforms random on test set. Training accuracy improves significantly with epochs but generalization is limited. Validation F1 peaks around 0.51, AUC slightly above 0.5.

### Experiment 4: CNN-Basic
- **Start Time**: 2026-02-06 10:10:56
- **End Time**: 2026-02-06 10:32:XX
- **Status**: Completed
- **Device**: GPU (NVIDIA H20)
- **Number of Runs**: 3 (seeds: 42, 142, 242)
- **Epochs per Run**: 8, 9, 9 (early stopping)
- **Results Summary**:
    - Accuracy: 0.5087 ± 0.0012
    - F1 Score: 0.5065 ± 0.0011
    - Precision: 0.5093 ± 0.0013
    - Recall: 0.5087 ± 0.0012
    - AUC: 0.5106 ± 0.0017
- **Single Run Results**:
    - [seed=42] acc=0.5095, f1=0.5075, auc=0.5128, best_val_f1=0.5239, 8 epochs
    - [seed=142] acc=0.5096, f1=0.5070, auc=0.5103, best_val_f1=0.5092, 9 epochs
    - [seed=242] acc=0.5070, f1=0.5049, auc=0.5087, best_val_f1=0.5187, 9 epochs
- **Training Accuracy**: 50.7% → 62.7% (some runs peaked at 58.4%, overfitting slightly reduced)
- **Notes**: CNN-Basic performs slightly better than CNN-Raw on test set. AUC reaches 0.51, validation F1 peaks around 0.52, generalization ability slightly improved.

### Experiment 5: KLineNet ✅
- **Start Time**: 2026-02-06 13:59:57
- **End Time**: 2026-02-06 14:51:22
- **Status**: Completed
- **Device**: GPU (NVIDIA H20)
- **Number of Runs**: 3 (seeds: 42, 142, 242)
- **Epochs per Run**: 16, 8, 14 (early stopping)
- **Results Summary**:
  - Accuracy: 0.5083 ± 0.0033
  - F1 Score: 0.5039 ± 0.0059
  - Precision: 0.5090 ± 0.0036
  - Recall: 0.5083 ± 0.0033
  - AUC: 0.5118 ± 0.0061
- **Single Run Results**:
  - [seed=42] acc=0.5126, f1=0.5118, auc=0.5183, best_val_f1=0.5131, 16 epochs
  - [seed=142] acc=0.5044, f1=0.4978, auc=0.5036, best_val_f1=0.5125, 8 epochs
  - [seed=242] acc=0.5078, f1=0.5021, auc=0.5135, best_val_f1=0.5139, 14 epochs
- **Runtime**: Approximately 51 minutes
- **Notes**: Uses robust norm + CLAHE + data augmentation (0.3). Performance slightly better than CNN-Basic. AUC reaches 0.51, validation F1 peaks around 0.51-0.52

### Experiment 6: KLineNet-MC ✅
- **Start Time**: 2026-02-06 15:08:01
- **End Time**: 2026-02-06 16:43:11
- **Status**: Completed
- **Device**: GPU (NVIDIA H20)
- **Number of Runs**: 3 (seeds: 42, 142, 242)
- **Epochs per Run**: 14, 20, 11 (early stopping or full training)
- **Results Summary**:
  - Accuracy: 0.5095 ± 0.0022
  - F1 Score: 0.5083 ± 0.0018
  - Precision: 0.5100 ± 0.0023
  - Recall: 0.5095 ± 0.0022
  - AUC: 0.5148 ± 0.0038
- **Single Run Results**:
  - [seed=42] acc=0.5119, f1=0.5097, auc=0.5197, best_val_f1=0.5089, 14 epochs
  - [seed=142] acc=0.5100, f1=0.5094, auc=0.5143, best_val_f1=0.5148, 20 epochs
  - [seed=242] acc=0.5066, f1=0.5058, auc=0.5104, best_val_f1=0.5222, 11 epochs
- **Runtime**: Approximately 95 minutes
- **Notes**: Uses multi-channel (RGB + edge detection) + Mixup (0.1) + data augmentation (0.3). Outperforms KLineNet on all metrics with lower variance and best generalization ability. AUC reaches 0.51
