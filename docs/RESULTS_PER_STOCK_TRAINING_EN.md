# Per-Stock Training Experiment

> **Experiment Status**: ✅ Pilot Test Completed (2026-02-07)
> **Conclusion**: ❌ Per-Stock method performs worse than mixed training, mainly due to insufficient single-stock data causing severe overfitting
> **Recommendation**: Not recommended for further expansion; consider transfer learning or grouped training approaches

---

## Experiment Motivation

The current mixed training approach (107 stocks combined) achieves approximately **50-51%** accuracy, close to random guessing. Possible reasons:
- Different stocks have varying volatility patterns (industry, market cap, liquidity differences)
- Mixed training causes the model to learn "average patterns", unable to capture individual stock characteristics
- Technical indicators such as support/resistance levels vary by stock

## Experiment Hypothesis

**Hypothesis: Training a dedicated model for each stock individually can better learn that stock's unique patterns and improve prediction accuracy**

**Experiment Conclusion**: ❌ **Hypothesis rejected**. Insufficient single-stock data volume (approximately 2,400 samples) leads to severe overfitting, with average performance (F1=45.67%) significantly lower than mixed training (F1=50.65%).

## Solution Comparison

| Dimension | Mixed Training (Existing) | Per-Stock Training (New Approach) |
|-----------|---------------------------|-----------------------------------|
| **Training Method** | 107 stocks mixed → 1 model | Each stock → 1 independent model |
| **Window Size** | 20 days | 60 days (longer time window) |
| **Training Samples** | 69,669 (mixed) | ~3,814/stock |
| **Advantages** | Large data volume, strong generalization | Personalized patterns, strong specificity |
| **Disadvantages** | Different stocks interfere with each other | Single-stock data scarce, prone to overfitting |
| **Expected Accuracy** | 51% | To be verified (expect >55%) |
| **Training Time** | 22 minutes (3 runs) | 10-15 hours (100 stocks × 3 runs) |

## Key Configuration Differences

### Special Design of [run_per_stock_experiments.py](../scripts/run_per_stock_experiments.py)

```python
CONFIG = {
    'window_size': 60,           # Longer window (20→60)
    'batch_size': 32,            # Smaller batch (128→32)
    'lr': 2e-4,                  # Smaller learning rate (4e-4→2e-4)
    'weight_decay': 1e-4,        # Stronger regularization (1e-5→1e-4)
    'patience': 3,               # More aggressive early stopping (5→3)
    'epochs': 15,                # Fewer epochs (20→15)
    'num_runs': 1,               # Run 1 time first (can change to 3)
}
```

**Design Principles:**
1. **Prevent Overfitting**: With limited single-stock data, use small learning rate + strong regularization + early stopping
2. **Adaptive Batch Size**: If stock data <200 samples, automatically reduce batch size
3. **Data Filtering**: Automatically skip stocks with insufficient data (<100 days)

## Experiment Workflow

### Phase 1: Pilot Test (30 minutes)

```bash
# Test 10 stocks to verify feasibility
python scripts/run_per_stock_experiments.py --num_stocks 10 --runs 1
```

**Expected Output:**
- Accuracy for each of the 10 stocks
- Average accuracy ± standard deviation
- Best/Worst 5 stocks ranking

**Evaluation Criteria:**
- If average accuracy > 52%: Approach is effective, continue expansion ✅
- If average accuracy < 52%: Approach is ineffective, needs adjustment ❌

### Phase 2: Medium Scale (2-3 hours)

```bash
# If Phase 1 is effective, expand to 50
python scripts/run_per_stock_experiments.py --num_stocks 50 --runs 1
```

### Phase 3: Full Experiment (10-15 hours)

```bash
# Full 100 stocks, run 3 times and average
python scripts/run_per_stock_experiments.py --num_stocks 100 --runs 3

# Support checkpoint resume (if interrupted)
python scripts/run_per_stock_experiments.py --num_stocks 100 --runs 3 --resume
```

## Performance Optimization

### 1. Checkpoint Resume
- Automatically save every 10 stocks trained
- If interrupted, use `--resume` to continue

### 2. Data Filtering
```python
# Automatically skip stocks with insufficient data
min_samples = window_size + prediction_horizon + 100
# For example: 60 + 5 + 100 = 165 days minimum requirement
```

### 3. Adaptive Configuration
```python
# Batch size automatically adjusted based on data volume
batch_size = min(32, train_size // 4)
# If training set only has 100 samples, batch=25
```

## Results Analysis

### Output Files

```
outputs/per_stock_results/
├── per_stock_results_10stocks.json    # Pilot results
├── per_stock_results_50stocks.json    # Medium scale
├── per_stock_results_100stocks.json   # Full experiment
└── models/                            # (Optional) Save model weights for each stock
    ├── AAPL_seed42.pt
    ├── 000001.SZ_seed42.pt
    └── ...
```

### Result Structure

```json
{
  "config": {...},
  "num_stocks": 10,
  "num_successful": 9,
  "num_failed": 1,
  "overall_avg": {
    "accuracy": 0.5234,
    "f1": 0.5189,
    "auc": 0.5267
  },
  "overall_std": {
    "accuracy": 0.0421,
    "f1": 0.0398,
    "auc": 0.0512
  },
  "individual_results": [
    {
      "stock_name": "AAPL",
      "success": true,
      "avg_metrics": {
        "accuracy": 0.5687,
        "f1": 0.5612,
        "auc": 0.5789
      }
    },
    ...
  ]
}
```

## Expected Discoveries

### Possible Outcome 1: Per-Stock Significantly Better

```
Mixed Training: 51.0% accuracy
Per-Stock Average: 55-58% accuracy ✅

→ Conclusion: Large differences in individual stock patterns, separate training is effective
→ Next Step: Analyze which stocks improved the most (possibly technically strong stocks)
```

### Possible Outcome 2: Per-Stock Slightly Better But Not Obvious

```
Mixed Training: 51.0%
Per-Stock Average: 52-53% ✅

→ Conclusion: Slight improvement, but limited by insufficient data volume
→ Next Step: Try transfer learning (pre-training + fine-tuning)
```

### Possible Outcome 3: Per-Stock Actually Worse

```
Mixed Training: 51.0%
Per-Stock Average: 48-50% ❌

→ Conclusion: Too little single-stock data, severe overfitting
→ Next Step: Consider grouped training (divide into 10 groups by industry/market)
```

## Visualization Analysis (Suggestions)

After experiment completion, can analyze:

1. **Accuracy Distribution Across Different Stocks**
   - Which stocks are easy to predict (>60%)?
   - Which stocks are completely random (~50%)?

2. **Relationship Between Accuracy and Stock Characteristics**
   - Market cap: Large-cap vs small-cap
   - Volatility: High volatility vs low volatility
   - Market: US vs CN
   - Industry: Tech vs Finance vs Manufacturing

3. **Training Curve Analysis**
   - Training set accuracy vs validation set accuracy
   - Overfitting? Is early stopping effective?

## Follow-up Directions

If per-stock is effective, can try:

1. **Transfer Learning**
   ```
   Step 1: Pre-train on all stocks
   Step 2: Fine-tune last few layers individually for each stock
   → Combine mixed training generalization + separate training personalization
   ```

2. **Grouped Training**
   ```
   Divide into 10 groups by industry/market:
   - Tech stocks → 1 model
   - Financial stocks → 1 model
   - ...
   → Balance data volume and personalization
   ```

3. **Ensemble Methods**
   ```
   For each stock:
   - Mixed model prediction: pred_mixed
   - Individual stock model prediction: pred_single
   - Weighted ensemble: α * pred_mixed + (1-α) * pred_single
   ```

## Usage Instructions

### Quick Start

```bash
# 1. Install dependencies (if not already installed)
pip install -r requirements.txt

# 2. Pilot test (recommended to run this first)
python scripts/run_per_stock_experiments.py --num_stocks 10

# 3. View results
cat outputs/per_stock_results/per_stock_results_10stocks.json | jq '.overall_avg'
```

### Parameter Description

```bash
--num_stocks N    # Train N stocks (default 10)
--runs R          # Run R times per stock (default 1, recommend 3)
--resume          # Checkpoint resume
```

### Modify Configuration

If you want to adjust training parameters, edit the `CONFIG` dictionary in the script:

```python
# scripts/run_per_stock_experiments.py line 35
CONFIG = {
    'window_size': 60,      # Try 90 for longer window
    'patience': 3,          # Change to 5 for more lenient
    'augment_prob': 0.2,    # Change to 0.5 for stronger data augmentation
    ...
}
```

## Notes

1. **GPU Memory**: Each stock trained independently, won't occupy large amounts of memory simultaneously
2. **Time Estimate**: 2-3 minutes/stock × N stocks × R runs
3. **Data Quality**: Will automatically skip stocks with insufficient data (<165 days)
4. **Randomness**: Single run may have fluctuations, recommend `--runs 3`

## Comparison Baseline

After completion, compare with existing results:

| Method | Accuracy | F1 Score | AUC | Training Time |
|--------|----------|----------|-----|---------------|
| LSTM (Mixed) | 0.4968 | 0.3298 | 0.5000 | 24 min |
| CNN-Basic (Mixed) | **0.5087** | **0.5065** | **0.5106** | 22 min |
| Per-Stock (New) | ❌ 0.5053±0.036 | ❌ **0.4567±0.075** | 0.5093±0.024 | 14 min/10 stocks |

**Goal: Exceed CNN-Basic's 0.5087 accuracy** → ❌ **Not achieved**

---

## ✅ Experiment Results (Pilot: 10 Stocks)

### Execution Information

- **Runtime**: 2026-02-07 06:02-06:16 (approx. 14 minutes)
- **Success Rate**: 10/10 (100%)
- **Configuration**: window_size=60, epochs=15, patience=3, runs=1
- **Dependency Issues Fixed**: Installed `mplfinance` and `libgl1-mesa-glx`

### Overall Performance

| Metric | Mean | Std Dev | vs CNN-Basic | Evaluation |
|--------|------|---------|--------------|------------|
| **Accuracy** | 0.5053 | ±0.0362 | -0.34% | ❌ Slightly lower |
| **F1 Score** | **0.4567** | **±0.0753** | **-9.83%** | ❌ **Significantly worse** |
| **Precision** | 0.5080 | ±0.1168 | - | - |
| **Recall** | 0.5053 | ±0.0362 | - | - |
| **AUC** | 0.5093 | ±0.0243 | -0.25% | ≈ Flat |

### Individual Stock Performance Ranking

#### 🏆 Top 5 Best Stocks

| Rank | Stock | Accuracy | F1 Score | AUC | Data Volume |
|------|-------|----------|----------|-----|-------------|
| 1 | 002153.SZ | 0.5282 | **0.5226** | **0.5499** | 4122 train |
| 2 | LLY | 0.5229 | **0.5220** | 0.5046 | 1374 train |
| 3 | TMO | 0.5154 | 0.5152 | 0.5236 | 2204 train |
| 4 | BX | 0.5137 | 0.5100 | 0.5064 | 4137 train |
| 5 | 002142.SZ | 0.5228 | 0.5067 | 0.5055 | 1824 train |

#### 💀 Bottom 5 Worst Stocks

| Rank | Stock | Accuracy | F1 Score | AUC | Data Volume |
|------|-------|----------|----------|-----|-------------|
| 1 | 600900.SS | 0.4212 | **0.3162** | 0.5170 | 2027 train |
| 2 | INTU | 0.4943 | **0.3270** | 0.5085 | 2450 train |
| 3 | 300760.SZ | 0.5596 | 0.4081 | 0.4636 | 2652 train |
| 4 | 000858.SZ | 0.4653 | 0.4612 | 0.4765 | 2667 train |
| 5 | 002241.SZ | 0.5100 | 0.4776 | 0.5373 | 2163 train |

### Key Observations

#### 1. ❌ **Severe Overfitting Problem**

Almost all stocks show a large performance gap between training and test sets:

| Stock | Train Acc | Test Acc | Overfitting Degree | Early Stop Epoch |
|-------|-----------|----------|-------------------|------------------|
| TMO | **90.1%** | 51.5% | -38.6% | 10 |
| 000858.SZ | **96.2%** | 46.5% | -49.7% | 14 |
| 002153.SZ | **88.7%** | 52.8% | -35.9% | 10 |
| 600900.SS | **84.1%** | 42.1% | -42.0% | 7 |
| INTU | **71.1%** | 49.4% | -21.7% | 4 |

**Conclusion**: Even with strong regularization (weight_decay=1e-4) and aggressive early stopping (patience=3), single-stock data volume (1374-4137 samples) is still insufficient to support generalization learning.

#### 2. 📊 **Huge Variance Between Individual Stocks**

- F1 Score standard deviation as high as **7.53%** (mixed training only 1.2%)
- Best stock (002153.SZ) f1=0.5226 ✅ **Exceeds mixed training**
- Worst stock (600900.SS) f1=0.3162 ❌ **Far below mixed training**

**Analysis**: Indicates that certain stocks do have learnable unique patterns, but overall average performance is dragged down by poor-performing stocks.

#### 3. 📉 **Insufficient Data Volume is the Core Problem**

| Method | Training Samples | F1 Score |
|--------|------------------|----------|
| Mixed Training | 69,669 | **0.5065** |
| Per-Stock (Average) | ~2,400 | 0.4567 |
| Data Volume Gap | **28x** | **-10% performance** |

### Corresponding Documentation Expectations

Experimental results match **"Possible Outcome 3"**:

> **Per-Stock actually worse (Mixed 51% vs Per-Stock 48-50%)**
> → Conclusion: Too little single-stock data, severe overfitting
> → Next Step: Consider grouped training (divide into 10 groups by industry/market)

---

## 💡 Experiment Conclusions and Recommendations

### ❌ Main Conclusions

1. **Per-Stock method not feasible with current data volume**
   - Average F1 Score decreased by **10%** (0.4567 vs 0.5065)
   - All stocks show severe overfitting
   - Large variance between individual stocks, overall performance unstable

2. **Not recommended to expand to 50 or 100 stocks**
   - Pilot results are clear: data volume is the bottleneck
   - Expansion would only waste computational resources (estimated 10-15 hours)
   - Unlikely to change conclusions

### ✅ Recommended Improvement Directions

#### Approach 1: **Transfer Learning** (Recommended Priority: ⭐⭐⭐⭐⭐)

```
Step 1: Pre-train on all 107 stocks (existing model)
Step 2: Fine-tune last 1-2 layers individually for each stock (few parameters)
Advantages: Combine mixed training generalization + separate training personalization
Expected improvement: 1-3%
```

#### Approach 2: **Grouped Training** (Recommended Priority: ⭐⭐⭐⭐)

```
Divide into 10-15 groups by industry/market:
- US Tech stocks → 1 model (~10 stocks)
- US Financial stocks → 1 model (~10 stocks)
- A股 Cyclical stocks → 1 model
- ...
Advantages: Balance data volume (~7,000 samples/group) and personalization
Expected improvement: 2-5%
```

#### Approach 3: **Expand Data Volume** (Recommended Priority: ⭐⭐⭐)

```
Current: 60-day window → ~2,400 samples/stock
Improvements:
- Use longer historical data (5 years → 10 years)
- Reduce window sliding step (currently 1 day → change to half-day)
- Use multi-scale windows (20 days + 40 days + 60 days)
Expected: Sample count increases 3-5x
```

#### Approach 4: **Ensemble Methods** (Recommended Priority: ⭐⭐⭐)

```
For each stock:
- Mixed model prediction: pred_mixed (f1=0.51)
- Individual stock model prediction: pred_single (f1=0.46, some stocks 0.52)
- Adaptive ensemble:
  - If individual model validation f1 > 0.52, weight 0.7
  - Otherwise use mixed model, weight 1.0
Expected improvement: 1-2%
```

---

## Appendix: Detailed Logs

Full results file: [outputs/per_stock_results/per_stock_results_10stocks.json](../outputs/per_stock_results/per_stock_results_10stocks.json)

### Example: Best Stock (002153.SZ)

```
Data volume: train=4122, val=974, test=975
Test performance: acc=0.5282, f1=0.5226, auc=0.5499
Training history:
  Epoch 1: train_acc=0.530 → val_f1=0.355
  Epoch 4: train_acc=0.683 → val_f1=0.470
  Epoch 7: train_acc=0.779 → val_f1=0.536 (best)
  Epoch 10: train_acc=0.887 → val_f1=0.506 (early stop triggered)
```

### Example: Worst Stock (600900.SS)

```
Data volume: train=2027, val=501, test=501
Test performance: acc=0.4212, f1=0.3162, auc=0.5170
Training history:
  Epoch 1: train_acc=0.527 → val_f1=0.467
  Epoch 4: train_acc=0.737 → val_f1=0.491 (best)
  Epoch 7: train_acc=0.841 → val_f1=0.422 (early stop triggered)
Severe overfitting: Training 84% → Testing 42%
```
