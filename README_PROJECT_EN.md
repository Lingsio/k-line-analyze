# K-Line Visual Pattern Recognition System

> Deep Learning-based K-Line Pattern Recognition and Stock Price Prediction System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Project Introduction

This project is a **deep learning-based K-line pattern similarity search and stock price prediction system**. The core innovation lies in converting stock K-line data into images and leveraging CNN image recognition capabilities to automatically learn price patterns, achieving:

- **🔍 Similar Pattern Search**: Fast matching of historical similar K-line patterns based on FAISS vector database
- **📈 Stock Trend Prediction**: CNN/Transformer models for predicting future price direction
- **🏭 Sector-Specific Models**: Specialized models trained by industry sector, significantly improving accuracy

### Core Results

| Metric | Result |
|------|------|
| **Average Accuracy** | **57.81%** (Target: 54%) |
| **Best Sector** | Consumer (60.37%) |
| **Stocks Covered** | 57 US stocks |
| **Industry Sectors** | 6 major sectors |

---

## 📖 Theoretical Framework

### Core Concept

> "Visual patterns in K-line charts contain effective information for predicting future price movements"

Traditional technical analysis relies on manual recognition of K-line patterns (doji, hammer, engulfing, etc.), which suffers from subjectivity and low efficiency. This project treats K-line charts as images and uses CNNs to automatically learn patterns:

```
OHLCV Data → Image Encoding → CNN Feature Extraction → Rise/Fall Prediction
```

### Encoding Methods

| Method | Description | Paper Reference |
|------|------|----------|
| **Candlestick** | Traditional candlestick charts | - |
| **OHLC Bars** | Sparse bar charts (recommended) | Xiu et al. (2021) |
| **GAF** | Gramian Angular Field | Chen & Tsai (2020) |
| **Hybrid** | Multi-channel fusion | - |

### Model Architectures

| Architecture | Input | Characteristics |
|------|------|------|
| **CNN (ResNet18)** | K-line images 256×256 | Image recognition, pretrained weights |
| **Transformer** | Sequences 10×9 | Self-attention, temporal modeling |
| **Grouped Models** | Trained by sector | Specialized optimization, heterogeneous processing |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/k-line-analyze.git
cd k-line-analyze

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download US stock data
python scripts/download_us_stocks.py

# Data storage location
data/raw/us/
├── AAPL.csv
├── MSFT.csv
└── ...
```

### Model Training

#### 1. Grouped Training (Recommended)

```bash
# Train all sectors (6 grouped models)
python scripts/train_grouped_h20.py --all --num-runs 3

# Train specific sector
python scripts/train_grouped_h20.py --sector Consumer
```

#### 2. Universal Models

```bash
# CNN v2 (image approach)
python scripts/train_cnn_predictor_v2.py --mode all --epochs 50

# Transformer v2 (sequence approach)
python scripts/train_transformer_predictor_v2.py --mode all --epochs 80

# Single-stock model
python scripts/train_cnn_predictor_v2.py --mode single --symbol AAPL
```

### Launch API Service

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for API documentation.

---

## 📊 Experimental Results

### SAK-Net Experimental Results (H20, 2026-02-11)

| Sector | Stocks | Accuracy | F1 | Status |
|------|--------|--------|-----|------|
| **Consumer** 🏆 | 10 | **60.37%** | 0.59 | ✅ Recommended for deployment |
| **Industrials_Energy** | 8 | **59.94%** | 0.52 | ✅ Recommended for deployment |
| **Tech_Semiconductors** | 6 | **59.88%** | 0.57 | ✅ Recommended for deployment |
| **Tech_Software** | 14 | **57.22%** | 0.58 | ✅ Deployable |
| **Financials** | 9 | **55.90%** | 0.55 | ⚠️ Needs optimization |
| **Healthcare** | 10 | **53.54%** | 0.52 | ⚠️ Needs optimization |
| **Average** | **57** | **57.81%** | - | ✅ Above target |

### Stock Coverage

**6 Major Sectors, 57 US Stocks:**

| Sector | Representative Stocks |
|------|----------|
| Tech_Semiconductors | NVDA, AMD, INTC, QCOM, TXN, AVGO |
| Tech_Software | MSFT, AAPL, GOOGL, META, AMZN, NFLX, TSLA... |
| Financials | JPM, GS, BLK, V, MA... |
| Healthcare | LLY, UNH, JNJ, PFE, ABBV... |
| Consumer | WMT, COST, MCD, DIS, NKE... |
| Industrials_Energy | BA, CAT, CVX, XOM, CSCO... |

---

## 🏗️ Project Structure

```
k-line-analyze/
├── core/                          # Core functional modules
│   ├── config.py                 # Configuration management
│   ├── data_fetcher.py           # Stock data acquisition
│   ├── similarity_search.py      # FAISS similarity search
│   └── utils/                    # Utility functions
│
├── models/                        # Model definitions
│   ├── cnn_encoder.py            # CNN encoder
│   ├── cnn_predictor_v2.py       # CNN predictor v2
│   ├── transformer_predictor.py  # Transformer predictor
│   └── prediction_dataset_v2.py  # Prediction dataset
│
├── src/                           # Extension modules
│   ├── data/
│   │   ├── image_generator.py    # Image generator (OHLC/GAF/Hybrid)
│   │   └── dataset.py            # Advanced datasets
│   └── models/
│       ├── cnn_model.py          # CNN model variants
│       └── lightweight_cnn.py    # Lightweight CNN
│
├── backend/                       # FastAPI backend service
│   └── app/
│       ├── main.py               # API entry point
│       ├── api/routes/           # API routes
│       └── services/             # Business services
│
├── scripts/                       # Training and experiment scripts
│   ├── train_grouped_h20.py      # Grouped training (H20)
│   ├── train_cnn_predictor_v2.py # CNN v2 training
│   ├── train_transformer_predictor_v2.py # Transformer training
│   └── run_v2_experiments.py     # V2 experiment runner
│
├── data/                          # Data directory
│   └── raw/us/                   # US stock data (CSV/Parquet)
│
├── outputs/                       # Output directory
│   ├── models/                   # Saved model weights
│   └── grouped_h20/              # Grouped training results
│
└── docs/                          # Documentation
    ├── THEORY.md                 # Detailed theoretical framework
    ├── RESULTS_SAK_NET_2026-02-11.md  # SAK-Net experiment records
    └── prediction-models.md      # Model usage instructions
```

---

## 🔬 Technical Highlights

### 1. Image Encoding Innovation

**OHLC Bars (Xiu et al. 2021)**
```python
# Sparse representation, 3 pixels per candlestick
- Open: horizontal line segment
- High/Low: vertical line segment
- Close: horizontal line segment
```

**GAF Encoding (Chen & Tsai 2020)**
```python
# Preserves temporal correlations
1. Normalize to [-1, 1]
2. Polar encoding: φ = arccos(x̃)
3. GASF: cos(φᵢ + φⱼ)
```

### 2. Grouped Training Strategy

```python
# Sector heterogeneity hypothesis
Tech_Semiconductors ≠ Healthcare  (different patterns)

# Independent training
Sector_Model = ResNet18(tech_stocks)  # Learn tech cycle patterns
Sector_Model = ResNet18(health_stocks)  # Learn drug trial events
```

**Result**: 51% → 58% (+7%)

### 3. Data Leakage Prevention

```python
# ❌ Wrong: filter entire dataset
train_ds = Dataset(quantile_filter=0.35)
val_ds = Dataset(quantile_filter=0.35)    # Leakage!
test_ds = Dataset(quantile_filter=0.35)   # Leakage!

# ✅ Correct: filter training set only
train_ds = Dataset(quantile_filter=0.35)  # Training only
val_ds = Dataset()                        # Full distribution
test_ds = Dataset()                       # Full distribution
```

---

## 📈 Performance Comparison

### Historical Evolution

| Stage | Method | Accuracy | Improvement |
|------|------|--------|------|
| Baseline | LSTM | 49.68% | - |
| Baseline | ResNet1D | 49.68% | 0% |
| CNN v1 | CNN (64×64) | 50.36% | +0.68% |
| CNN v2 | CNN (128×128) | 50.87% | +0.51% |
| + CLAHE | KLineNet | 50.83% | -0.04% |
| + MultiChannel | KLineNet-MC | 50.95% | +0.12% |
| **Grouped** | **Grouped Training** | **57.81%** | **+6.86%** 🚀 |

### Key Findings

1. **Grouped Training** is the most effective improvement method (+7%)
2. **OHLC Encoding** outperforms traditional candlestick charts
3. **20-day window** + **128px** is the optimal balance
4. **ResNet18** + **pretrained weights** is critical

---

## 🛠️ Advanced Configuration

### H20 (96GB VRAM) Configuration

```python
H20_CONFIG = {
    'batch_size': 128,
    'num_workers': 8,
    'epochs': 50,
    'lr': 4e-4,
    'use_amp': True,        # Mixed precision
    'use_compile': True,    # torch.compile
}
```

### RTX 4060 (8GB VRAM) Configuration

```python
RTX4060_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'epochs': 50,
    'lr': 1e-3,
    'model': 'lightweight_cnn',  # 500K parameters
}
```

---

## 📚 Documentation Navigation

| Document | Content |
|------|------|
| [THEORY.md](THEORY.md) | Complete theoretical framework, mathematical principles, technical implementation |
| [METHOD_IMAGE_ENCODING.md](METHOD_IMAGE_ENCODING.md) | Image encoding methods (OHLC/GAF/Hybrid) |
| [METHOD_MULTISCALE_FUSION.md](METHOD_MULTISCALE_FUSION.md) | Multi-scale CNN fusion methods |
| [METHOD_OPTIMIZED_PIPELINE.md](METHOD_OPTIMIZED_PIPELINE.md) | 6-stage optimization pipeline |
| [docs/RESULTS_SAK_NET_2026-02-11.md](docs/RESULTS_SAK_NET_2026-02-11.md) | **Latest experiment records (57.81%)** |
| [docs/USER_GUIDE_PREDICTION_MODELS.md](docs/USER_GUIDE_PREDICTION_MODELS.md) | Model usage instructions, API documentation |

---

## 🤝 Contribution Guidelines

### Development Workflow

```bash
# 1. Fork the repository
# 2. Create branch
git checkout -b feature/your-feature

# 3. Commit changes
git commit -m "feat: add your feature"

# 4. Push branch
git push origin feature/your-feature

# 5. Create Pull Request
```

### Code Standards

- **Naming**: lowercase+underscore (files), PascalCase (classes), snake_case (functions)
- **Type annotations**: Required for important functions
- **Documentation**: Google-style docstrings

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Xiu et al. (2021)** - OHLC bar chart design
- **Chen & Tsai (2020)** - GAF encoding method
- **PyTorch Team** - Deep learning framework
- **FAISS Team** - Vector search engine

---

## 📞 Contact Information

- Project Homepage: https://github.com/your-org/k-line-analyze
- Issue Reporting: https://github.com/your-org/k-line-analyze/issues
- Email: your-email@example.com

---

> ⚠️ **Disclaimer**: This project is for academic research purposes only and does not constitute investment advice. The stock market involves risks; invest with caution.
