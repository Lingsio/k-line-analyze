"""Run single group training - for debugging"""
import os, sys, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

group_stocks = ["BA", "GE", "HON", "CAT", "CVX", "XOM", "COP", "CSCO"]
group_name = "Industrials_Energy"

config = {
    'img_size': (128, 128),
    'norm_method': 'robust',
    'use_clahe': True,
    'output_channels': 'rgb+edge',
    'label_threshold': 'dynamic',
    'augment_prob': 0.3,
    'mixup_prob': 0.1,
    'batch_size': 64,
    'lr': 4e-4,
    'weight_decay': 1e-5,
    'patience': 5,
    'epochs': 20,
}

print(f"Loading data for {group_name}...", flush=True)
print(f"Stocks: {group_stocks}", flush=True)

common = {
    'data_dir': str(PROJECT_ROOT / 'data' / 'raw' / 'us'),
    'window_size': 20, 'prediction_horizon': 5,
    'img_size': config['img_size'], 'norm_method': config['norm_method'],
    'use_clahe': config['use_clahe'], 'output_channels': config['output_channels'],
    'label_threshold': config['label_threshold'],
}

# Load full datasets
train_ds_full = StockDataset(mode='train', augment_prob=config['augment_prob'],
                             mixup_prob=config['mixup_prob'], **common)
val_ds_full = StockDataset(mode='val', augment_prob=0.0, **common)
test_ds_full = StockDataset(mode='test', augment_prob=0.0, **common)

print(f"Full dataset loaded: train={len(train_ds_full)}, val={len(val_ds_full)}, test={len(test_ds_full)}", flush=True)

# Filter by group stocks
stock_set = set(group_stocks)
train_indices = [i for i, sample in enumerate(train_ds_full.samples) if sample['ticker'] in stock_set]
val_indices = [i for i, sample in enumerate(val_ds_full.samples) if sample['ticker'] in stock_set]
test_indices = [i for i, sample in enumerate(test_ds_full.samples) if sample['ticker'] in stock_set]

print(f"Filtered indices: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}", flush=True)

# Check which stocks are actually in the dataset
train_stocks = set(train_ds_full.samples[i]['ticker'] for i in train_indices)
print(f"Stocks found in train set: {train_stocks}", flush=True)

val_stocks = set(val_ds_full.samples[i]['ticker'] for i in val_indices)
print(f"Stocks found in val set: {val_stocks}", flush=True)

test_stocks = set(test_ds_full.samples[i]['ticker'] for i in test_indices)
print(f"Stocks found in test set: {test_stocks}", flush=True)

# Check which stocks are in data_cache
print(f"\nStocks in train_ds_full.data_cache: {list(train_ds_full.data_cache.keys())}", flush=True)
