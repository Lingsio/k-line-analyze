# 3-Class Prediction with Neutral Category

## Overview

This document describes the implementation of 3-class stock prediction (Up/Neutral/Down) with special evaluation rules to prevent the model from being "lazy" and always predicting neutral.

## Key Features

### 1. Three-Class Labels
- **0 (Down)**: Return < -threshold
- **1 (Neutral)**: |Return| <= threshold
- **2 (Up)**: Return > threshold

### 2. Data Leakage Prevention

**CRITICAL**: Filtering is ONLY applied to the training set.

```python
# Training set: can use filtering
train_ds = StockDataset(
    mode='train',
    quantile_filter=0.35,          # Only for training
    train_filter_threshold=0.003,  # Only for training
    ...
)

# Val/Test set: NO filtering (keep all samples)
val_ds = StockDataset(mode='val', ...)   # No filtering
test_ds = StockDataset(mode='test', ...) # No filtering
```

### 3. Special Evaluation: Neutral Penalty

The key insight: We want the model to be confident in its predictions. Predicting "neutral" when the actual move is significant counts as **WRONG**.

```python
# Evaluation logic:
if predicted_neutral and actual_return > threshold:
    count_as_wrong()  # Should have predicted Up
elif predicted_neutral and actual_return < -threshold:
    count_as_wrong()  # Should have predicted Down
```

## Usage

### Using the Pipeline (Stage 7)

```bash
# Run stage 7 only
python scripts/run_optimized_pipeline.py --stage 7

# Debug mode
python scripts/run_optimized_pipeline.py --stage 7 --debug

# Run all stages including 7
python scripts/run_optimized_pipeline.py --stage all
```

### Using the Standalone Script

```bash
# Default settings (1% threshold)
python scripts/train_3class_predictor.py

# Custom threshold
python scripts/train_3class_predictor.py --threshold 0.015 --epochs 50

# Debug mode
python scripts/train_3class_predictor.py --debug
```

### Using in Your Own Code

```python
from src.data.dataset import StockDataset
from torch.utils.data import DataLoader

# Configuration
config = {
    'window_size': 20,
    'prediction_horizon': 5,
    'img_size': (128, 128),
    'label_threshold': 0.01,    # 1% threshold for neutral zone
    'num_classes': 3,           # 3-class
    'chart_type': 'candle',
}

# Training set WITH filter
train_ds = StockDataset(
    data_dir='data/raw/us',
    mode='train',
    train_filter_threshold=0.003,  # Filter extreme neutrals during training
    **config
)

# Val/Test WITHOUT filter (avoid data leakage!)
val_ds = StockDataset(data_dir='data/raw/us', mode='val', **config)
test_ds = StockDataset(data_dir='data/raw/us', mode='test', **config)

# Create loaders
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_classes` | Number of classes (2 or 3) | 2 |
| `label_threshold` | Threshold for neutral zone | 0.01 (1%) |
| `quantile_filter` | Filter middle X% of returns (train only) | None |
| `train_filter_threshold` | Filter samples with \|return\| < X (train only) | None |

## Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall correct predictions
- **F1 (weighted)**: Weighted F1 score across all classes
- **Per-class Recall**: Recall for Down/Neutral/Up separately

### Strict Metrics (with Neutral Penalty)
- **Strict Accuracy**: Counts "predicted neutral but actual up/down" as WRONG
- **Neutral Wrong Rate**: Percentage of incorrect neutral predictions

## Example Output

```
3-Class Evaluation (neutral threshold=1.00%):
  Standard Accuracy: 0.5243
  Strict Accuracy: 0.4892
  Neutral Wrong Rate: 0.0351
  F1 (weighted): 0.5187
  
Class distribution:
  Predicted: Down=125, Neutral=342, Up=133
  Actual: Down=198, Neutral=201, Up=201
```

## Design Rationale

### Why Filter Training Only?

1. **Avoid Data Leakage**: If we filter val/test, we're not evaluating on the true distribution
2. **Real-world Distribution**: In production, we see ALL market conditions, not just "clear signal" ones
3. **Unbiased Metrics**: Val/Test metrics should reflect real-world performance

### Why Neutral Penalty?

Without this penalty, the model learns to:
- Predict neutral for uncertain samples (safe choice)
- Achieve high accuracy by being conservative
- Fail to capture actual market movements

With the penalty, the model must:
- Make directional predictions when there's signal
- Only predict neutral when truly uncertain
- Be accountable for missed opportunities

## Tips

1. **Threshold Selection**: Start with 1% (0.01). Adjust based on:
   - Market volatility (higher volatility → higher threshold)
   - Prediction horizon (longer horizon → higher threshold)

2. **Training Filter**: Start with 0.3% (0.003). This removes extreme noise while keeping most samples.

3. **Class Imbalance**: 3-class often results in imbalanced data. Use class weights:
   ```python
   class_weights = train_ds.get_class_weights()
   criterion = nn.CrossEntropyLoss(weight=class_weights)
   ```

4. **Monitoring**: Watch both standard and strict accuracy. A large gap indicates the model is over-predicting neutral.
