# Image Encoding Methods - Implementation Summary

## Implemented Improvements

### 1. OHLC Bar Representation (Xiu et al. 2021) ✅

**File**: `src/data/image_generator.py`

**New Methods**:
- `draw_ohlc_bars()` - Renders OHLC bar charts
- Each trading day occupies 3 pixels width (open, high-low, close)
- Black background with white bars (sparse representation, beneficial for CNN)
- All stock prices normalized to a unified scale

**Key Features**:
- Design consistent with Xiu et al. paper
- Normalization enables comparison across different stocks
- Preserves volume information (bottom 20% of the image)

### 2. GAF Encoding (Chen & Tsai 2020) ✅

**File**: `src/data/image_generator.py`

**New Methods**:
- `create_gaf_ohlc()` - Encodes OHLCV data as GAF images
- Supports GASF (Gramian Angular Summation Field) and GADF
- Multi-channel output (Open, High, Low, Close, Volume)

**Mathematical Principles**:
```
1. Normalize to [-1, 1]: x̃ = (2x - max - min) / (max - min)
2. Polar encoding: φ = arccos(x̃)
3. GASF: cos(φ_i + φ_j)
4. GADF: sin(φ_i - φ_j)
```

**Key Features**:
- Preserves temporal dependencies
- Captures relative correlations
- Diagonal contains original value information

### 3. Short Window + High Resolution ✅

**Configuration**:
- Window size: 10 days (vs original 20 days)
- Image size: 256×256 (vs original 128×128)
- Candle/bar width: ~25 pixels (vs original ~6 pixels)

**Advantages**:
- Clearer K-line patterns
- Ability to distinguish details like doji and hammer patterns
- Better alignment with experimental settings in referenced papers

## New Experimental Configurations

### Experiment Scripts

**File**: `scripts/run_v2_experiments.py`

**New Experiments**:
1. `OHLC-10d-256` - Pure OHLC bar chart
2. `GAF-10d-256` - Pure GAF encoding
3. `OHLC-GAF-10d-256` - OHLC + GAF hybrid
4. `Hybrid-10d-256` - Fused representation
5. `Candle-20d-128` - Baseline comparison

### Usage

```bash
# Run all V2 experiments
python scripts/run_v2_experiments.py --experiment all --num-runs 3

# Run single experiment
python scripts/run_v2_experiments.py --experiment OHLC-10d-256

# Visual comparison
python scripts/visualize_representations.py --symbol AAPL
```

## Dataset Updates

**File**: `src/data/dataset.py`

**New Parameters**:
- `chart_type`: 'candle', 'ohlc', 'gaf', 'hybrid'
- `use_gaf`: Enable GAF augmentation
- `gaf_method`: 'gasf' or 'gadf'

**Example**:
```python
dataset = StockDataset(
    data_dir='data/raw/us',
    window_size=10,          # 10-day window
    img_size=(256, 256),     # High resolution
    chart_type='ohlc',       # OHLC bar chart
    use_gaf=True,            # Enable GAF
    gaf_method='gasf',
)
```

## Comparison with Original Version

| Feature | Original | V2 Improvement |
|---------|----------|----------------|
| Chart Type | Candlestick | Candlestick / OHLC / GAF / Hybrid |
| Window Size | 20 days | 10 days (configurable) |
| Image Size | 128×128 | 256×256 (configurable) |
| Candle/Bar Width | ~6 pixels | ~25 pixels |
| Normalization | MinMax/Robust | Added percentile-based normalization |
| Data Augmentation | Basic | Added time warping, data scaling |
| Multi-scale | Single-scale | Multi-channel fusion supported |

## Expected Results

Based on results from reference papers:

| Method | Expected Accuracy | Reference Source |
|--------|-------------------|------------------|
| Original CNN | 51.8% | Current baseline |
| OHLC 10d/256 | 54-56% | Xiu et al. (2021) - 53%+ |
| GAF Encoding | 55-58% | Chen & Tsai (2020) - 90.7% (pattern recognition) |
| Hybrid Method | 56-60% | Combining advantages of both |

## Implementation Details

### OHLC Bar Chart Rendering
```python
def draw_ohlc_bars(self, open_p, high_p, low_p, close_p, volume, ...):
    # Each trading day = 3 pixels
    # Pixel 0: Open horizontal line
    # Pixel 1: High-low vertical line
    # Pixel 2: Close horizontal line
    # Price normalized to image height
```

### GAF Encoding
```python
def create_gaf_ohlc(self, open_p, high_p, low_p, close_p, volume, ...):
    # 1. Normalize to [-1, 1]
    # 2. Polar conversion: φ = arccos(x̃)
    # 3. Calculate GASF: cos(φ_i + φ_j)
    # 4. Output 5 channels (O, H, L, C, V)
```

## Training Recommendations

1. **Batch Size**: Due to larger 256×256 images, recommend batch_size=64 (vs original 128)
2. **Learning Rate**: 3e-4 (slightly lower than original 4e-4)
3. **Training Epochs**: 30 epochs (more data augmentation requires more epochs to converge)
4. **Early Stopping Patience**: 7 epochs (prevent overfitting)

## File Change List

### Modified Files
1. `src/data/image_generator.py` - Added OHLC, GAF, Hybrid methods
2. `src/data/dataset.py` - Added chart_type, use_gaf parameters
3. `src/models/cnn_model.py` - Optimized ResNet18 for larger inputs
4. `scripts/run_baseline_experiments.py` - Added V2 experiment configurations

### New Files
1. `scripts/run_v2_experiments.py` - V2 experiment dedicated script
2. `scripts/visualize_representations.py` - Visualization comparison tool
3. `V2_IMPROVEMENTS.md` - This document

## Next Steps

1. **Run V2 Experiments**:
   ```bash
   python scripts/run_v2_experiments.py --experiment all
   ```

2. **Compare Results**:
   - Compare OHLC vs GAF vs Hybrid methods
   - Check if overfitting issues are mitigated
   - Verify cross-stock generalization capability

3. **Further Optimization**:
   - If OHLC performs well, try Multi-Scale OHLC
   - If GAF performs well, try MTF (Markov Transition Field)
   - Adjust fusion weights for GAF and OHLC

## References

1. **Xiu et al. (2021)** - "(Re-)Imag(in)ing Price Trends"
   - CNN extracts signals from price charts with 53%+ accuracy
   - OHLC bar charts outperform traditional candlestick charts
   - Importance of cross-stock normalization

2. **Chen & Tsai (2020)** - "Encoding candlesticks as images"
   - GAF-CNN achieves 90.7% accuracy on pattern recognition
   - GAF preserves temporal dependencies and correlations
   - Applicable to 8 classic K-line patterns

3. **Duong et al. (2025)** - "Investigating Market Strength Prediction"
   - Candlestick pattern detection does not help improve performance
   - Pure CNN learns more effectively from raw images
   - Time series data outperforms image conversion
