# K-Line Visual Pattern Search

A Deep Learning-based K-line (candlestick) pattern similarity search tool. Converts K-line (candlestick) charts into searchable vector embeddings, enabling fast pattern similarity retrieval.

## Core Features

- **K-line Image Generation**: Renders OHLCV data into standardized K-line images
- **Feature Extraction**: Uses CNN/ResNet to extract vector representations of K-line patterns
- **Similarity Search**: Fast pattern matching based on vector distance
- **DTW Matching**: Dynamic Time Warping for sequence similarity calculation

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: PyTorch, torchvision, numpy, pandas, scikit-learn, matplotlib, mplfinance

## Project Structure

```
k-line-photo/
├── core/                    # Core Functionality Modules
│   ├── config.py           # Configuration Management
│   ├── data_fetcher.py     # Data Fetching
│   ├── dataset.py          # Dataset Definitions
│   ├── preprocessor.py     # Data Preprocessing
│   ├── similarity_search.py # Similarity Search
│   ├── dtw_matcher.py      # DTW Matching
│   ├── feature_extractor.py # Feature Extraction
│   └── utils/              # Utility Functions
│       ├── kline_renderer.py  # K-line Rendering
│       └── gaf_transformer.py # GAF Transformation
├── models/                  # Model Definitions
│   ├── cnn_encoder.py      # CNN Encoder (Main Model)
│   └── baselines.py        # Baseline Models (LSTM, ResNet1D)
├── src/                     # Extension Modules
│   ├── data/               # Data Processing
│   │   ├── data_loader.py  # Data Loading
│   │   ├── dataset.py      # Advanced Datasets
│   │   └── image_generator.py # Image Generator
│   └── models/             # Model Variants
│       ├── cnn_model.py    # CNN Model
│       └── rnn_model.py    # RNN Model
├── scripts/                 # Data Download Scripts
│   ├── download_cn_stocks.py    # Download A-Share Data
│   ├── download_cn_baostock.py  # BaoStock Data Source
│   └── download_us_stocks.py    # Download US Stock Data
└── data/raw/               # Raw Data Storage
```

## Quick Start

### 1. Download Data

```bash
# Download A-Share (Chinese stocks) data
python scripts/download_cn_baostock.py

# Download US stock data
python scripts/download_us_stocks.py
```

### 2. Basic Usage

```python
from core.preprocessor import Preprocessor
from core.dataset import KLineDataset
from models.cnn_encoder import CNNEncoder

# Load data
preprocessor = Preprocessor()
data = preprocessor.load_stock_data("data/raw/AAPL.csv")

# Create dataset
dataset = KLineDataset(data, window_size=60)

# Load model
model = CNNEncoder(embedding_dim=256)
```

### 3. Generate K-line Images

```python
from src.data.image_generator import ImageGenerator

generator = ImageGenerator(
    figsize=(128, 128),
    normalization='minmax'
)
image = generator.generate(ohlcv_data)
```

### 4. Similarity Search

```python
from core.similarity_search import SimilaritySearch

searcher = SimilaritySearch(model)
similar_patterns = searcher.search(query_pattern, k=10)
```

## Data Format

CSV files contain the following columns:
- `date`: Date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## Configuration Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `WINDOW_SIZE` | 60 | Input days |
| `IMAGE_SIZE` | 128 | Image resolution |
| `EMBEDDING_DIM` | 256 | Embedding vector dimension |
| `BATCH_SIZE` | 32 | Batch size |

## Documentation Navigation

### Method Documentation (METHOD_)

| Document | Content |
|----------|---------|
| [THEORY.md](./THEORY.md) | Complete theoretical framework and mathematical principles |
| [METHOD_IMAGE_ENCODING.md](./METHOD_IMAGE_ENCODING.md) | Image encoding methods (OHLC/GAF/Hybrid) |
| [METHOD_MULTISCALE_FUSION.md](./METHOD_MULTISCALE_FUSION.md) | Multi-scale CNN fusion methods |
| [METHOD_OPTIMIZED_PIPELINE.md](./METHOD_OPTIMIZED_PIPELINE.md) | 6-stage optimized pipeline guide |
| [docs/METHOD_3CLASS_CLASSIFICATION.md](./docs/METHOD_3CLASS_CLASSIFICATION.md) | Three-class classification (Rise/Flat/Fall) |
| [docs/METHOD_RTX4060_TRAINING.md](./docs/METHOD_RTX4060_TRAINING.md) | RTX 4060 training guide |

### Experiment Results (RESULTS_)

| Document | Content |
|----------|---------|
| [RESULTS_BASELINE_EXPERIMENTS.md](./RESULTS_BASELINE_EXPERIMENTS.md) | Baseline experiments comparison (CNN/LSTM/ResNet1D) |
| [docs/RESULTS_EXPERIMENTS_COMPARISON.md](./docs/RESULTS_EXPERIMENTS_COMPARISON.md) | Full experiment comparison summary |
| [docs/RESULTS_SAK_NET_2026-02-11.md](./docs/RESULTS_SAK_NET_2026-02-11.md) | **SAK-Net Experiments (Latest Results: 57.81%)** |
| [docs/RESULTS_GROUPED_TRAINING.md](./docs/RESULTS_GROUPED_TRAINING.md) | Grouped training method details |
| [docs/RESULTS_PER_STOCK_TRAINING.md](./docs/RESULTS_PER_STOCK_TRAINING.md) | Per-stock training experiments |

### User Guides

| Document | Content |
|----------|---------|
| [README_PROJECT.md](./README_PROJECT.md) | Complete project introduction and quick start |
| [docs/USER_GUIDE_PREDICTION_MODELS.md](./docs/USER_GUIDE_PREDICTION_MODELS.md) | Prediction model API usage guide |

## License

MIT License
