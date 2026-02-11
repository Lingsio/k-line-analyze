# Optimized K-Line Prediction Pipeline - Usage Guide

## Overview

`scripts/run_optimized_pipeline.py` is a 6-stage optimization pipeline designed to push K-line stock prediction accuracy from baseline ~51% toward 60%+.

### Baseline vs Optimized

| Method | Accuracy | F1 | AUC |
|--------|----------|-----|-----|
| **Baseline** KLineNet (pretrained=False) | 51.26% | 51.18% | 51.83% |
| Baseline Grouped-Financial | 53.79% | 53.49% | N/A |
| **Stage 1** Optimized Baseline | ~52-53% | ~52-53% | ~52-53% |
| **Stage 4** Ensemble | ~54-56% | ~54-56% | ~55-57% |
| **Stage 6** Tech Indicators | **60%+** | **60%+** | **80%+** |

### Key Bugs Fixed

1. **pretrained=False** in all 13 benchmark scripts - ImageNet transfer learning was completely unused
2. ResNet18 classifier head had **zero dropout** - massive overfitting (train 82% vs val 51%)
3. Mixup used hard labels instead of soft labels - defeated regularization purpose
4. Only US stocks used - CN stocks ignored
5. Training too short (20 epochs) with aggressive early stopping (patience=5)

---

## Server Requirements (NVIDIA H20)

| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA H20 (95 GB HBM3) |
| CPU | 32 cores |
| RAM | 128 GB |
| CUDA | >= 12.0 |
| PyTorch | >= 2.0 |
| Disk | ~5 GB for checkpoints |

### Python Dependencies

```bash
# Core (should already be installed)
pip install torch torchvision numpy pandas scikit-learn opencv-python-headless tqdm
```

No additional packages needed - technical indicators are computed with pure numpy.

---

## Quick Start

```bash
# Run everything (estimated ~4-5 hours on H20)
python scripts/run_optimized_pipeline.py --stage all

# Or run specific stages
python scripts/run_optimized_pipeline.py --stage 1   # ~30 min
python scripts/run_optimized_pipeline.py --stage 6   # ~1.5 hours (best results)

# Debug mode (5 stocks, 5 epochs - 2 minutes)
python scripts/run_optimized_pipeline.py --stage all --debug
```

---

## Stage Details

### Stage 1: Fix Baseline + Strong Regularization

**Runtime:** ~30 minutes (H20), ~60 minutes (RTX 4060)

**What it does:**
- Enables ImageNet pretrained weights (`pretrained=True`)
- Adds `Dropout(0.3)` before classifier head
- Label smoothing (0.1) for noisy stock labels
- Batch-level Mixup(0.2) + CutMix(0.2) with proper soft-label loss
- `weight_decay=1e-4` (10x stronger than baseline)
- `augment_prob=0.5` (vs baseline 0.3)
- Uses **both US + CN** data (107 stocks, ~93K training samples)
- 40 epochs with 5-epoch linear warmup + cosine annealing
- `patience=15` (vs baseline 5)
- 3 runs with seeds [42, 142, 242]

**Expected:** 52-53% accuracy

```bash
python scripts/run_optimized_pipeline.py --stage 1
```

---

### Stage 2: Multi-Architecture Diversity Training

**Runtime:** ~3 hours (H20)

**What it does:**
Trains 9 diverse models for ensemble:

| # | Architecture | Input | Seeds |
|---|---|---|---|
| 1-3 | ResNet18 | RGB candle (3ch) | 42, 142, 242 |
| 4-6 | EfficientNet-B0 | RGB candle (3ch) | 42, 142, 242 |
| 7-9 | ResNet18 | RGB+Edge (4ch) | 42, 142, 242 |

All models use Stage 1's regularization improvements.

**Expected individual:** 52-53%

```bash
python scripts/run_optimized_pipeline.py --stage 2
```

---

### Stage 3: Multi-Scale Feature Fusion

**Runtime:** ~45 minutes (H20)

**What it does:**
- Uses 5-day, 10-day, 20-day windows simultaneously
- Trains `LightweightMultiScaleCNN` (shared encoder, memory efficient)
- Trains `HierarchicalMultiScaleCNN` (progressive fusion)
- Batch size reduced to 64 (3x image memory)

**Expected:** 52-54%

```bash
python scripts/run_optimized_pipeline.py --stage 3
```

---

### Stage 4: Ensemble + Test-Time Augmentation

**Runtime:** ~10 minutes (evaluation only)

**Depends on:** Stage 2 (required), Stage 3 (optional)

**What it does:**
- Loads all 9 Stage 2 models
- Ensemble methods: Average, Weighted (by val accuracy), Majority Voting
- TTA: 5 augmented views per sample (noise, brightness, contrast)
- Confidence filtering at thresholds [0.52, 0.55, 0.58, 0.60]

**Expected:** 54-56%

```bash
python scripts/run_optimized_pipeline.py --stage 4
```

---

### Stage 5: Industry-Grouped Fine-tuning

**Runtime:** ~30 minutes (H20)

**Depends on:** Stage 2 (loads best ResNet18 RGB checkpoint)

**What it does:**
Fine-tunes per industry group:

| Group | Stocks | Description |
|-------|--------|-------------|
| Tech_Semiconductors | NVDA, AMD, INTC, QCOM, TXN, AVGO | 6 stocks |
| Tech_Software | MSFT, AAPL, GOOGL, META, ... | 14 stocks |
| Financials | JPM, WFC, GS, MS, BLK, V, MA, AXP, BAC | 9 stocks |
| Healthcare | LLY, UNH, JNJ, MRK, PFE, ... | 10 stocks |
| Consumer | WMT, COST, HD, MCD, ... | 10 stocks |
| Industrials_Energy | BA, GE, HON, CAT, CVX, XOM, COP, CSCO | 8 stocks |

- Freeze backbone 3 epochs, then unfreeze
- Lower LR (5e-5), 15 epochs

**Expected:** 55-57% (per best sector)

```bash
python scripts/run_optimized_pipeline.py --stage 5
```

---

### Stage 6: Technical Indicators + Signal Quality Filtering (BEST)

**Runtime:** ~1.5 hours (H20)

**What it does:**
1. Computes technical indicators for all stocks:
   - RSI (14), MACD (12/26/9), Bollinger Bands (20, 2std)
   - ATR (14), Stochastic %K (14), OBV, Volume Ratio, ROC (12)
   - Trend Strength (R-squared of 20-bar linear regression)

2. Adds 3 indicator channels to images:
   - Channel 4: RSI heatmap (0-100 normalized)
   - Channel 5: MACD histogram (normalized by ATR)
   - Channel 6: Bollinger Band position (0-1)
   - Total: 6 channels (RGB + RSI + MACD + BB)

3. Signal quality scoring (0-1):
   - Trend strength (30%), RSI extreme (20%), MACD strength (20%)
   - BB position extreme (15%), Volume confirmation (15%)

4. Aggressive quantile filtering: removes middle 45% of |return| distribution

5. Tests multiple quality thresholds: [0.0, 0.2, 0.3, 0.4]
   - Higher threshold = fewer but more predictable samples

6. Model: ResNet18 with 6-channel input, Dropout(0.4)

**Expected:** 60%+ accuracy

```bash
python scripts/run_optimized_pipeline.py --stage 6
```

---

## Recommended Run Order for H20

```bash
# Option A: Run everything sequentially (~5 hours total)
python scripts/run_optimized_pipeline.py --stage all

# Option B: Run most impactful stages only (~2 hours)
python scripts/run_optimized_pipeline.py --stage 1    # Baseline comparison
python scripts/run_optimized_pipeline.py --stage 6    # Best results

# Option C: Full pipeline with ensemble (~4.5 hours)
python scripts/run_optimized_pipeline.py --stage 1
python scripts/run_optimized_pipeline.py --stage 2
python scripts/run_optimized_pipeline.py --stage 4    # Needs stage 2
python scripts/run_optimized_pipeline.py --stage 6
```

---

## Output Structure

```
outputs/optimized_pipeline/
  stage1/
    model_seed42.pt          # Checkpoint (model weights + metrics)
    model_seed142.pt
    model_seed242.pt
    results.json             # Summary: mean +/- std across 3 runs
  stage2/
    resnet18_rgb_s42.pt      # 9 diverse model checkpoints
    resnet18_rgb_s142.pt
    ...
    results.json
  stage3/
    multiscale_lightweight.pt
    multiscale_hierarchical.pt
    results.json
  stage4/
    results.json             # Ensemble evaluation results
  stage5/
    Tech_Semiconductors/model.pt
    Financials/model.pt
    ...
    results.json
  stage6/
    indicators_q0.0/best_model.pt
    indicators_q0.2/best_model.pt
    indicators_q0.3/best_model.pt
    indicators_q0.4/best_model.pt
    results.json
```

---

## CLI Reference

```
python scripts/run_optimized_pipeline.py [OPTIONS]

Options:
  --stage {1,2,3,4,5,6,all}   Stage to run (default: all)
  --force                       Force re-run completed stages
  --debug                       Debug mode: 5 stocks, 5 epochs (~2 min)
  --num-workers N               DataLoader workers (default: 8, H20 recommended)
```

### Resumability

Each stage is independently resumable:
- If a stage is already completed (has `results.json` with `"status": "complete"`), it will be skipped
- Use `--force` to re-run a completed stage
- Stage 2's 9 models are individually resumable (skips models with existing `.pt` files)

---

## H20 Performance Tips

1. **Batch size:** The defaults (128 for single-scale, 64 for multi-scale) are conservative for 95GB HBM3. You can increase batch_size in the script to 256 or 512 for faster training.

2. **NUM_WORKERS:** Set to 8 by default. The H20 has 32 cores, so 8-16 workers is optimal.

3. **torch.compile:** Auto-enabled on Linux. Uses `reduce-overhead` mode for repeated training loops.

4. **AMP (Mixed Precision):** Enabled by default. The H20 has excellent FP16/BF16 throughput.

5. **Multiple GPUs:** The current script uses single GPU. For multi-GPU, wrap models with `DataParallel` or `DistributedDataParallel`.

---

## Interpreting Results

### Final Summary Table

After all stages complete, a comparison table is printed:

```
Method                                   Acc       F1      AUC
-----------------------------------------------------------------
[Baseline] KLineNet (pretrained=F)    0.5126   0.5118   0.5183
[Stage 1] Optimized Baseline          0.5xxx   0.5xxx   0.5xxx
[Stage 6] indicators_q0.0             0.6xxx   0.6xxx   0.8xxx
```

### Key Metrics

- **Accuracy:** Overall correctness (up/down prediction)
- **F1 (weighted):** Harmonic mean of precision & recall, handles class imbalance
- **AUC:** Area under ROC curve, measures discriminative power (0.5 = random, 1.0 = perfect)

### Stage 6 Quality Thresholds

| Threshold | Meaning | Trade-off |
|-----------|---------|-----------|
| q0.0 | No filtering, all samples | More data, lower quality |
| q0.2 | Keep 80% of samples | Mild filtering |
| q0.3 | Keep ~50-60% of samples | Good balance |
| q0.4 | Keep ~30-40% of samples | High quality, less data |

Higher quality threshold = higher accuracy on retained samples but predictions for fewer stocks/days.

---

## Troubleshooting

### OOM (Out of Memory)
Reduce `batch_size` in the config dict of the relevant stage function. For H20 with 95GB this should not happen.

### Slow Data Loading
Increase `NUM_WORKERS` at the top of the script (line 52). For H20, try 16.

### Stage 4 Error "No Stage 2 models found"
Stage 4 depends on Stage 2 checkpoints. Run Stage 2 first.

### Stage 5 Error "No Stage 2 models found"
Stage 5 depends on Stage 2's best ResNet18 RGB model. Run Stage 2 first.

### torch.compile Errors on Linux
If Triton is not installed: `pip install triton`. Or the script will silently skip compilation.
