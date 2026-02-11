# Stock Prediction Model Usage Guide

## Overview

This project introduces two types of **stock price trend prediction models** built on top of the existing K-Line similarity search system. Both support "all stocks" and "single stock" training modes. Currently focused on the **US market**.

| Model | Input | Core Idea |
|------|------|----------|
| **CNN (k-line-net-mc)** | 10-day K-line chart 256×256 RGB | Image recognition of candlestick patterns |
| **Transformer** | 10-day × 9-dimensional numerical sequence | Self-attention for capturing temporal dependencies |

### Prediction Categories (Binary Classification)

| Label | Value | Condition |
|------|---|------|
| **Down** | 0 | Future closing price < Current closing price |
| **Up** | 1 | Future closing price >= Current closing price |

Accuracy = Number of correctly predicted samples / Total samples, intuitively straightforward.

### Prediction Time Horizons

By default, predicts price movements for **T+1, T+3, T+5** days (customizable via `--horizons`).

---

## File Structure

```
backend/app/models/
├── cnn_predictor_v2.py         # CNN prediction model (v2, 10-day/256px, binary)
├── transformer_predictor.py    # Transformer prediction model (binary)
├── prediction_dataset_v2.py    # Dataset + temporal split
├── cnn_predictor.py            # CNN v1 (legacy)
└── prediction_dataset.py       # Dataset v1

scripts/
├── train_cnn_predictor_v2.py          # CNN v2 training script ← Recommended
├── train_transformer_predictor_v2.py  # Transformer v2 training script ← Recommended
├── train_cnn_predictor.py              # CNN v1 training script
└── train_transformer_predictor.py      # Transformer v1 training script

backend/app/api/routes/
├── prediction_v2.py            # v2 prediction API
└── prediction.py               # v1 prediction API
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train Models

#### CNN v2 (Recommended — Image Recognition Approach)

```bash
# Train on all US stocks (universal model)
python scripts/train_cnn_predictor_v2.py --mode all --epochs 50

# Train on single stock (specialized model)
python scripts/train_cnn_predictor_v2.py --mode single --symbol AAPL --market us

# Single stock — NVDA
python scripts/train_cnn_predictor_v2.py --mode single --symbol NVDA --market us --epochs 80
```

#### Transformer v2

```bash
# Train on all US stocks
python scripts/train_transformer_predictor_v2.py --mode all --epochs 80

# Train on single stock
python scripts/train_transformer_predictor_v2.py --mode single --symbol NVDA --market us

# GPU optimization (for high VRAM cards like H20)
python scripts/train_transformer_predictor_v2.py --mode all \
    --batch-size 256 --d-model 256 --nhead 8 --layers 6 --amp
```

### 3. Model Output Locations

```
backend/trained_models/predictors/
├── cnn_v2_universal.pt                # CNN universal model
├── cnn_v2_us_AAPL.pt                  # CNN single stock model
├── transformer_v2_universal.pt        # Transformer universal model
├── transformer_v2_us_NVDA.pt          # Transformer single stock model
├── *_history.json                     # Training loss/accuracy curves
└── *_config.json                      # Model configuration parameters
```

---

## Training Parameters

### CNN v2 Parameters

| Parameter | Default | Description |
|------|--------|------|
| `--mode` | `all` | `all` = All stocks, `single` = Single stock |
| `--symbol` | - | Stock ticker (required for single mode) |
| `--market` | `us` | Market (currently focused on US) |
| `--window-size` | `10` | Number of K-line days per image |
| `--image-size` | `256` | Image resolution |
| `--horizons` | `1 3 5` | Prediction time horizons (days) |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--freeze-backbone` | `5` | Epochs to freeze ResNet backbone |
| `--val-ratio` | `0.15` | Validation set ratio |
| `--start-date` | 8 years ago | Data start date (YYYY-MM-DD) |
| `--end-date` | Today | Data end date (YYYY-MM-DD) |

### Transformer v2 Additional Parameters

| Parameter | Default | Description |
|------|--------|------|
| `--d-model` | `128` | Transformer hidden dimension |
| `--nhead` | `8` | Number of attention heads |
| `--layers` | `4` | Number of Transformer encoder layers |
| `--dim-ff` | `512` | Feed-forward layer dimension |
| `--dropout` | `0.1` | Dropout rate |
| `--warmup-epochs` | `5` | Learning rate warmup epochs |
| `--amp` | `false` | Enable mixed precision training (FP16) |

---

## API Usage

Start the backend:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

### Endpoints

| Endpoint | Method | Description |
|------|------|------|
| `/api/v1/prediction/v2/models` | GET | List all trained v2 models |
| `/api/v1/prediction/v2/predict` | POST | Predict using specified model |
| `/api/v1/prediction/v2/compare/{symbol}` | GET | Compare predictions from all v2 models |

### Prediction Request Examples

```bash
# Transformer universal model predicting AAPL
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "market": "us", "model_type": "transformer"}'

# CNN single stock model predicting AAPL
curl -X POST http://localhost:8000/api/v1/prediction/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "market": "us", "model_type": "cnn", "model_variant": "cnn_v2_us_AAPL"}'

# Compare all models
curl http://localhost:8000/api/v1/prediction/v2/compare/TSLA?market=us
```

### Response Format

```json
{
  "symbol": "AAPL",
  "market": "us",
  "model_type": "transformer",
  "window_size": 10,
  "image_size": 256,
  "predictions": {
    "T+1": {
      "predicted_class": "Up",
      "class_probabilities": {
        "Down": 0.35,
        "Up": 0.65
      },
      "predicted_return": 0.008
    },
    "T+3": { "..." : "..." },
    "T+5": { "..." : "..." }
  }
}
```

---

## Design Rationale

### Why Binary Classification (Up/Down) Instead of Multi-class?

- **Accuracy intuition**: 50% is the random baseline, anything above indicates predictive ability
- **Class balance**: Up/Down days are roughly balanced, no need for complex class weighting
- **Practicality**: The core investment decision is "buy or not buy"

### Why v2 Uses 10 Days + 256×256?

| Item | v1 (60-day/128px) | v2 (10-day/256px) |
|------|-----------------|-----------------|
| Candlestick width | ~2px (body 1px) | **~25px (body 19px)** |
| Pattern distinguishability | Cannot distinguish patterns | Doji, Hammer, Engulfing patterns clearly visible |
| ResNet first layer 7×7 coverage | ~3-4 days | **~2-3 candles** (meaningful local patterns) |

### Why Use Temporal Split for Validation?

Random splitting would scatter adjacent windows into training and validation sets (98% overlap), inflating validation metrics.
Temporal splitting ensures all validation data comes after training data, simulating the real "predict future from history" scenario.

### Universal Model vs Single Stock Model

| | Universal Model | Single Stock Model |
|---|---|---|
| Training data | 26 US stocks | Single stock only |
| Generalization | Strong, can predict unseen stocks | Weak, only applicable to that stock |
| Specialization | General | May better capture stock-specific patterns |
| Recommended use | Default | Additional training for stocks of interest |

---

## Built-in US Stock List (26 Tickers)

```
AAPL  MSFT  GOOGL  AMZN  NVDA  TSLA  META  AMD
NFLX  INTC  JPM    BAC   V     MA    JNJ   PFE
UNH   PG    KO     WMT   XOM   CVX   BA    DIS
CRM   CSCO
```

Covering five sectors: Technology, Financials, Healthcare, Consumer, and Energy.
