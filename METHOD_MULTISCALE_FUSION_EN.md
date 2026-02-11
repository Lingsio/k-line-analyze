# V2 Complete Improvement Plan Summary

## All Improvements Implemented ✅

### 1. Image Representation Improvements ✅

#### OHLC Bar Chart (`src/data/image_generator.py`)
- **Method**: `draw_ohlc_bars()`
- **Features**:
  - 3-pixel width per trading day (open, high-low, close)
  - Black background + white bars (sparse representation)
  - Based on Xiu et al. (2021) paper

#### GAF Encoding (`src/data/image_generator.py`)
- **Method**: `create_gaf_ohlc()`
- **Features**:
  - Gramian Angular Summation Field (GASF)
  - Preserves temporal dependencies and correlations
  - Based on Chen & Tsai (2020) paper

#### Configuration
- Window: 10 days (vs 20 days)
- Resolution: 256×256 (vs 128×128)
- Each candlestick: ~25 pixels wide (vs ~6 pixels)

---

### 2. Multi-Scale Feature Fusion ✅

#### File: `src/models/multiscale_cnn.py`

**Three Architectures**:

1. **MultiScaleKLineEncoder**
   - Independent ResNet18 encoders processing 5/10/20 days
   - Cross-scale self-attention mechanism
   - Learnable scale weights

2. **HierarchicalMultiScaleCNN**
   - Progressive fusion (5d→10d→20d)
   - Hierarchical feature extraction

3. **LightweightMultiScaleCNN**
   - Shared-weight encoder
   - Multi-task output heads
   - Soft ensemble learning

#### File: `src/data/multiscale_dataset.py`

- Simultaneously generates 5/10/20 day window images
- Maintains aligned starting indices
- Supports all chart types

---

### 3. Advanced Labeling Strategies ✅

#### File: `src/data/advanced_labeling.py`

**Four Strategies**:

1. **Volatility (Volatility-Adjusted)**
   ```python
   threshold = 0.5 * vol * sqrt(horizon)
   ```
   - Adaptive threshold based on historical volatility
   - Filters sideways market samples

2. **Quantile**
   - Based on historical return distribution
   - Top/bottom quantiles define up/down
   - Middle region is sideways

3. **Regime (Market Regime)**
   - Detects trending/choppy/high-vol/low-vol states
   - Different thresholds for different states
   - Trending markets require larger moves

4. **Consensus (Multi-Timeframe Consensus)**
   - Consensus voting from 1/3/5/10 day predictions
   - High-confidence signals

**Utility Classes**:
- `SmartThresholdCalculator`: Pre-calculates thresholds per stock
- `filter_extreme_samples`: Filters outlier samples
- `balance_classes`: Class balancing

---

### 4. Transfer Learning ✅

#### File: `scripts/train_transfer_learning.py`

**Two-Stage Training**:

1. **Pre-training Stage**
   ```bash
   python scripts/train_transfer_learning.py --stage pretrain --markets us cn
   ```
   - Trains universal model using all market data
   - Learns generic candlestick patterns
   - High learning rate (1e-3)

2. **Fine-tuning Stage**
   ```bash
   python scripts/train_transfer_learning.py --stage finetune --target-market us
   ```
   - Loads pre-trained weights
   - Freezes backbone for initial epochs
   - Low learning rate (1e-4)
   - Optimizes for specific market/stock

**Key Parameters**:
- `--freeze-epochs`: Number of epochs to freeze backbone
- `--lr-finetune`: Fine-tuning learning rate (typically 1/10 of pre-training)

---

### 5. Ensemble Learning ✅

#### File: `scripts/train_ensemble.py`

**Two Ensemble Strategies**:

1. **Diverse Ensemble**
   - Different architectures (ResNet18, EfficientNet)
   - Different chart types (OHLC, GAF, Hybrid)
   - Different random seeds
   - Learnable ensemble weights

2. **Multi-Scale Ensemble**
   - Combines three multi-scale models
   - Attention + Hierarchical + Lightweight

**Ensemble Methods**:
- Average Ensemble
- Weighted Ensemble (weighted by validation accuracy)
- Voting Ensemble (Majority Voting)

**Usage**:
```bash
python scripts/train_ensemble.py --type diverse --num-models 5
python scripts/train_ensemble.py --type multiscale
python scripts/train_ensemble.py --type both
```

---

## Quick Start

### 1. Basic Experiments (OHLC + GAF)

```bash
python scripts/run_v2_experiments.py --experiment all --num-runs 3
```

### 2. Complete V2 Pipeline

```bash
# Run all stages
python scripts/run_complete_v2.py --stage all

# Run specific stage only
python scripts/run_complete_v2.py --stage baseline
python scripts/run_complete_v2.py --stage multiscale
```

### 3. Transfer Learning

```bash
# Full pipeline
python scripts/train_transfer_learning.py --stage both \
    --markets us cn \
    --target-market us \
    --epochs-pretrain 30 \
    --epochs-finetune 20
```

### 4. Ensemble Models

```bash
# Train ensemble
python scripts/train_ensemble.py --type both --num-models 5
```

### 5. Visualization

```bash
# Compare different representations
python scripts/visualize_representations.py --symbol AAPL
```

---

## Expected Performance Improvements

| Method | Expected Accuracy | Improvement |
|--------|------------------|-------------|
| Original (CNN-Basic) | 51.8% | - |
| OHLC 10d/256 | 54-56% | +2-4% |
| GAF Encoding | 55-58% | +3-6% |
| Multi-Scale Fusion | 56-60% | +4-8% |
| + Transfer Learning | 57-61% | +5-9% |
| + Ensemble Models | 58-63% | +6-11% |

---

## File List

### New Files

```
src/models/
├── multiscale_cnn.py          # Multi-scale CNN models (3 architectures)

src/data/
├── multiscale_dataset.py       # Multi-scale dataset
├── advanced_labeling.py        # Advanced labeling strategies

scripts/
├── run_v2_experiments.py       # V2 basic experiments
├── run_complete_v2.py          # Complete V2 pipeline
├── train_transfer_learning.py  # Transfer learning
├── train_ensemble.py           # Ensemble learning
├── visualize_representations.py # Visualization tool

V2_IMPROVEMENTS.md              # Initial improvement documentation
V2_COMPLETE_SUMMARY.md          # This document
```

### Modified Files

```
src/data/
├── image_generator.py          # + OHLC, GAF, Hybrid methods
└── dataset.py                  # + chart_type, use_gaf parameters

src/models/
└── cnn_model.py                # Optimized ResNet18 for 256x256

scripts/
└── run_baseline_experiments.py # + V2 experiment configuration
```

---

## Key Hyperparameter Recommendations

### Basic Training
- `window_size`: 10
- `img_size`: (256, 256)
- `batch_size`: 64 (256x256 images are larger)
- `lr`: 3e-4
- `epochs`: 30

### Multi-Scale Training
- `scales`: [5, 10, 20]
- `batch_size`: 32 (higher memory requirements)
- `lr`: 2e-4

### Transfer Learning
- Pre-training `lr`: 1e-3
- Fine-tuning `lr`: 1e-4
- `freeze_epochs`: 5

### Ensemble Learning
- `num_models`: 3-5
- Different `seed`, `chart_type`, `arch`

---

## Future Optimization Directions

1. **Hyperparameter Search**: Automated tuning with Optuna
2. **Larger Dataset**: Add more international market data
3. **Attention Visualization**: Understand which candlestick patterns the model focuses on
4. **Online Learning**: Model self-adapts over time
5. **Multi-Task Learning**: Simultaneously predict direction + volatility + trend strength

---

## References

1. **Xiu et al. (2021)** - "(Re-)Imag(in)ing Price Trends"
   - OHLC bar charts, multi-scale, transfer learning

2. **Chen & Tsai (2020)** - "Encoding candlesticks as images"
   - GAF encoding, time series to image conversion

3. **Duong et al. (2025)** - "Investigating Market Strength Prediction"
   - Pure CNN outperforms pattern detection
