# RTX 4060 Grouped Training Guide

A grouped training solution optimized for NVIDIA RTX 4060 8GB VRAM.

## Features

- **Lightweight CNN**: Only 500K parameters (1/20 of ResNet18)
- **VRAM Optimization**: Batch Size 32 requires only ~4-6GB VRAM
- **Sector Grouping**: Independent training for 6 major sectors
- **3-Class Support**: Up/Neutral/Down classification

## Quick Start

### 1. Environment Test

```bash
python scripts/test_4060_setup.py
```

This detects GPU, tests model VRAM usage, and verifies the training loop.

### 2. Train All Sectors

```bash
# 3-class mode (recommended)
python scripts/train_grouped_4060.py --all

# 2-class mode
python scripts/train_grouped_4060.py --all --num-classes 2

# Use smaller model
python scripts/train_grouped_4060.py --all --variant ultra
```

### 3. Train Single Sector

```bash
# View all sectors
python scripts/train_grouped_4060.py --sector Tech_Software

# Available sectors:
# - Tech_Semiconductors
# - Tech_Software
# - Financials
# - Healthcare
# - Consumer
# - Industrials_Energy
```

### 4. Debug Mode

```bash
# Quick test (train only 5 epochs per sector)
python scripts/train_grouped_4060.py --all --debug
```

## Model Comparison

| Model | Parameters | VRAM Usage (batch=32) | Recommended Scenario |
|-------|------------|----------------------|---------------------|
| UltraLight | 100K | ~2-3GB | Extremely limited VRAM |
| Light | 500K | ~4-6GB | **Recommended** |
| ResNet18 | 11M | ~6-8GB | Large VRAM GPU |

## Key Parameters

### Data Parameters
- `window_size`: 20 days (lookback period)
- `prediction_horizon`: 5 days (prediction period)
- `label_threshold`: 1% (neutral zone threshold)

### Training Parameters
- `batch_size`: 32 (recommended for 4060)
- `epochs`: 50 (early stopping patience=10)
- `lr`: 0.001
- `dropout`: 0.3

## Data Leakage Prevention

Key design: **Filtering is only applied to the training set**

```python
# Training set: filtering allowed
train_ds = StockDataset(
    mode='train',
    train_filter_threshold=0.003,  # Only for training
    ...
)

# Validation/Test sets: no filtering (preserve true distribution)
val_ds = StockDataset(mode='val', ...)   # No filtering
test_ds = StockDataset(mode='test', ...) # No filtering
```

## Output Structure

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

## Performance Expectations

Based on RTX 4060 8GB:
- Training time per sector: ~10-30 minutes (depending on data volume)
- Total training time: ~2-3 hours (6 sectors)
- Expected accuracy: 50-55% (financial prediction is challenging)

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/train_grouped_4060.py --all --batch-size 16

# Use ultra model
python scripts/train_grouped_4060.py --all --variant ultra
```

### Slow Training

```bash
# Increase workers (based on your CPU cores)
# Edit scripts/train_grouped_4060.py:
# RTX4060_CONFIG['num_workers'] = 8
```

### Insufficient Samples

Some sectors may have fewer stocks; training will be skipped if samples < 50.

## Advanced Usage

### Custom Parameters

Edit `RTX4060_CONFIG` in `scripts/train_grouped_4060.py`:

```python
RTX4060_CONFIG = {
    'batch_size': 32,
    'window_size': 20,          # Change to 10 or 30
    'prediction_horizon': 5,    # Change to 1 (next day) or 10
    'label_threshold': 0.01,    # Change to 0.015 (1.5%)
    'num_classes': 3,           # 2 or 3
    'epochs': 50,
    'lr': 1e-3,
    'dropout': 0.3,
}
```

### Loading Trained Models

```python
import torch
from src.models.lightweight_cnn import build_lightweight_cnn

# Load
checkpoint = torch.load('outputs/grouped_models_4060/Tech_Software_3class.pt')
config = checkpoint['config']

# Rebuild model
model = build_lightweight_cnn(
    variant=config['cnn_variant'],
    num_classes=config['num_classes']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Test metrics
print(f"Test Accuracy: {checkpoint['test_metrics']['accuracy']:.4f}")
print(f"Test F1: {checkpoint['test_metrics']['f1']:.4f}")
```

## Next Steps

1. Run `test_4060_setup.py` to verify environment
2. Run `train_grouped_4060.py --all --debug` for quick test
3. Train formally with `train_grouped_4060.py --all`
4. View results at `outputs/grouped_models_4060/summary.json`
