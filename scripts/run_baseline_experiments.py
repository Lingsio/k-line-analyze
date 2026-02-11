"""
Full ECCV Experiments - Complete Version (Optimized for NVIDIA H20 + 32-core CPU)
运行时间预估：30-60分钟 (优化后)
数据：50 US + 50 CN 股票 (107 stocks total)
每个模型运行3次以获得统计显著性

Optimizations:
- Mixed Precision Training (AMP) for faster GPU computation
- Multi-worker data loading with pin_memory
- Larger batch sizes to utilize GPU memory
- torch.compile for model optimization
- Persistent workers to reduce data loading overhead

Usage:
    python scripts/run_full_eccv_experiments.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path
import multiprocessing

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel
from models.baselines import LSTMEncoder, ResNet1DEncoder

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIRS = [str(PROJECT_ROOT / 'data' / 'raw' / 'us')]  # Only US stocks
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eccv_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware-optimized config for NVIDIA H20 + 32-core CPU + 128GB RAM
NUM_WORKERS = min(8, multiprocessing.cpu_count())  # 8 workers for data loading
PIN_MEMORY = torch.cuda.is_available()  # Faster GPU transfer
USE_AMP = torch.cuda.is_available()  # Mixed precision training
USE_COMPILE = hasattr(torch, 'compile')  # PyTorch 2.0+ model compilation
PREFETCH_FACTOR = 4  # Prefetch batches per worker

# Full experiment config - optimized for H20 GPU (96GB HBM3)
CONFIG = {
    'epochs': 20,
    'batch_size': 128,  # Increased from 32 (H20 has large memory)
    'lr': 4e-4,  # Scale learning rate with batch size (linear scaling)
    'weight_decay': 1e-5,
    'patience': 5,
    'num_runs': 3,  # 3 runs for statistical significance
    'gradient_accumulation': 1,  # Can increase if OOM
}

# All experiments to run
EXPERIMENTS = {
    # Sequence baselines (CPU to avoid NaN)
    'LSTM': {
        'model_type': 'lstm',
        'device': 'cpu',
    },
    'ResNet1D': {
        'model_type': 'resnet1d',
        'device': 'cpu',
    },
    # CNN baselines
    'CNN-Raw': {
        'model_type': 'cnn',
        'img_size': (64, 64),
        'norm_method': 'minmax',
        'use_clahe': False,
        'output_channels': 'rgb',
        'label_threshold': 0.005,
        'augment_prob': 0.0,
        'device': 'cuda',
    },
    'CNN-Basic': {
        'model_type': 'cnn',
        'img_size': (128, 128),
        'norm_method': 'minmax',
        'use_clahe': False,
        'output_channels': 'rgb',
        'label_threshold': 0.005,
        'augment_prob': 0.0,
        'device': 'cuda',
    },
    # Our methods
    'KLineNet': {
        'model_type': 'cnn',
        'img_size': (128, 128),
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'device': 'cuda',
    },
    'KLineNet-MC': {
        'model_type': 'cnn',
        'img_size': (128, 128),
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb+edge',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'mixup_prob': 0.1,
        'device': 'cuda',
    },
    # =========================================================================
    # NEW V2 Methods - Based on Research Papers
    # 1. OHLC Bar Charts (Xiu et al. 2021)
    # 2. GAF Encoding (Chen & Tsai 2020)
    # 3. 10-day window + 256x256 resolution
    # =========================================================================
    'OHLC-10d-256': {
        'model_type': 'cnn',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'ohlc',
        'norm_method': 'robust',
        'use_clahe': False,  # OHLC is already sparse/clean
        'output_channels': 'rgb',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'device': 'cuda',
    },
    'GAF-10d-256': {
        'model_type': 'cnn',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'gaf',
        'gaf_method': 'gasf',
        'norm_method': 'robust',
        'use_clahe': False,
        'output_channels': 'rgb',
        'label_threshold': 'dynamic',
        'augment_prob': 0.2,
        'device': 'cuda',
    },
    'OHLC-GAF-Hybrid': {
        'model_type': 'cnn',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'ohlc',
        'use_gaf': True,
        'gaf_method': 'gasf',
        'norm_method': 'robust',
        'use_clahe': False,
        'output_channels': 'rgb',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'device': 'cuda',
    },
    'Hybrid-10d-256': {
        'model_type': 'cnn',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'hybrid',
        'norm_method': 'robust',
        'use_clahe': False,
        'output_channels': 'rgb',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'device': 'cuda',
    },
}

# ============================================================================
# Training Functions
# ============================================================================

def create_dataloaders(config, model_type):
    """Create train/val/test dataloaders."""
    common_args = {
        'data_dir': DATA_DIRS,
        'prediction_horizon': 5,
    }
    
    # Support for new V2 configs (10-day window, chart types)
    common_args['window_size'] = config.get('window_size', 20)

    if model_type == 'cnn':
        common_args.update({
            'img_size': config.get('img_size', (128, 128)),
            'norm_method': config.get('norm_method', 'robust'),
            'use_clahe': config.get('use_clahe', True),
            'output_channels': config.get('output_channels', 'rgb'),
            'label_threshold': config.get('label_threshold', 'dynamic'),
            # New V2 parameters
            'chart_type': config.get('chart_type', 'candle'),
            'use_gaf': config.get('use_gaf', False),
            'gaf_method': config.get('gaf_method', 'gasf'),
        })

    train_ds = StockDataset(
        mode='train',
        augment_prob=config.get('augment_prob', 0.0),
        mixup_prob=config.get('mixup_prob', 0.0),
        **common_args
    )
    val_ds = StockDataset(mode='val', augment_prob=0.0, **common_args)
    test_ds = StockDataset(mode='test', augment_prob=0.0, **common_args)

    # Optimized DataLoader settings for high-performance hardware
    loader_kwargs = {
        'batch_size': CONFIG['batch_size'],
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY,
        'prefetch_factor': PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        'persistent_workers': NUM_WORKERS > 0,  # Keep workers alive between epochs
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def create_model(config, input_channels, device):
    """Create model based on config."""
    model_type = config['model_type']

    if model_type == 'cnn':
        model = CNNModel(
            num_classes=2,
            input_channels=input_channels,
            arch='resnet18',
            pretrained=False
        )
    elif model_type == 'lstm':
        model = LSTMEncoder(
            input_dim=2,
            hidden_dim=128,
            num_layers=2,
            embedding_dim=256,
            num_classes=2
        )
    elif model_type == 'resnet1d':
        model = ResNet1DEncoder(
            input_dim=2,
            layers=[2, 2, 2, 2],
            embedding_dim=256,
            num_classes=2
        )

    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device, model_type, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    use_amp = scaler is not None and device.type == 'cuda'

    for batch in loader:
        imgs, seqs, labels = batch
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # Mixed precision forward pass
        with autocast(device_type='cuda', enabled=use_amp):
            if model_type == 'cnn':
                inputs = imgs.to(device, non_blocking=True)
                outputs = model(inputs)
            else:
                seqs = seqs.to(device, non_blocking=True)
                if model_type == 'lstm':
                    _, outputs = model(seqs)
                else:  # resnet1d
                    seqs = seqs.permute(0, 2, 1)
                    _, outputs = model(seqs)

            loss = criterion(outputs, labels)

        # Skip NaN loss
        if torch.isnan(loss):
            continue

        # Mixed precision backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    return total_loss / max(len(loader), 1), acc


def evaluate(model, loader, criterion, device, model_type):
    """Evaluate model with optional mixed precision."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    use_amp = USE_AMP and device.type == 'cuda'

    with torch.no_grad():
        for batch in loader:
            imgs, seqs, labels = batch
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=use_amp):
                if model_type == 'cnn':
                    inputs = imgs.to(device, non_blocking=True)
                    outputs = model(inputs)
                else:
                    seqs = seqs.to(device, non_blocking=True)
                    if model_type == 'lstm':
                        _, outputs = model(seqs)
                    else:
                        seqs = seqs.permute(0, 2, 1)
                        _, outputs = model(seqs)

                loss = criterion(outputs, labels)

            if not torch.isnan(loss):
                total_loss += loss.item()

            probs = torch.softmax(outputs.float(), dim=1)  # Cast to float for softmax
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = {
        'loss': total_loss / max(len(loader), 1),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
    }

    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc'] = 0.5

    return metrics


def run_single_experiment(name, config, seed=42):
    """Run a single experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = config.get('device', 'cuda')
    device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    model_type = config['model_type']

    print(f"  [seed={seed}] Device: {device}", flush=True)

    # Create data
    print(f"  [seed={seed}] Loading data...", flush=True)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders(config, model_type)
    print(f"  [seed={seed}] Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)

    # Create model
    input_channels = train_ds.get_num_channels() if model_type == 'cnn' else 2
    model = create_model(config, input_channels, device)

    # Compile model for faster execution (PyTorch 2.0+)
    if USE_COMPILE and device.type == 'cuda' and model_type == 'cnn':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print(f"  [seed={seed}] Model compiled with torch.compile", flush=True)
        except Exception as e:
            print(f"  [seed={seed}] torch.compile failed: {e}", flush=True)

    # Training setup
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # Mixed precision scaler
    scaler = GradScaler() if USE_AMP and device.type == 'cuda' else None

    # Training loop
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    history = []

    print(f"  [seed={seed}] Training (AMP={scaler is not None})...", flush=True)
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, model_type, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type)
        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
        })

        print(f"  [seed={seed}] Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
              f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}", flush=True)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"  [seed={seed}] Early stopping at epoch {epoch+1}", flush=True)
                break

    # Load best model and evaluate on test
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_metrics = evaluate(model, test_loader, criterion, device, model_type)

    print(f"  [seed={seed}] Test: acc={test_metrics['accuracy']:.4f}, "
          f"f1={test_metrics['f1']:.4f}, auc={test_metrics['auc']:.4f}", flush=True)

    return {
        'seed': seed,
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'epochs_trained': len(history),
        'history': history,
    }


def run_experiment_with_runs(name, config, num_runs=3):
    """Run experiment multiple times for statistical significance."""
    print(f"\n{'='*70}", flush=True)
    print(f"Experiment: {name}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Config: {config}", flush=True)

    all_runs = []
    for run_id in range(num_runs):
        seed = 42 + run_id * 100
        print(f"\n--- Run {run_id+1}/{num_runs} ---", flush=True)
        result = run_single_experiment(name, config, seed=seed)
        all_runs.append(result)

    # Aggregate results
    metrics_keys = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    summary = {}
    for key in metrics_keys:
        values = [r['test_metrics'][key] for r in all_runs]
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))

    print(f"\n=== {name} Summary ===", flush=True)
    print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}", flush=True)
    print(f"F1 Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}", flush=True)
    print(f"AUC:      {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}", flush=True)

    return {
        'name': name,
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                   for k, v in config.items()},
        'runs': all_runs,
        'summary': summary,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70, flush=True)
    print("ECCV Full Experiments - K-Line Visual Pattern Recognition", flush=True)
    print("Optimized for NVIDIA H20 + 32-core CPU + 128GB RAM", flush=True)
    print("="*70, flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    print(f"CPU Workers: {NUM_WORKERS}", flush=True)
    print(f"Mixed Precision (AMP): {USE_AMP}", flush=True)
    print(f"Model Compilation: {USE_COMPILE}", flush=True)
    print(f"Batch Size: {CONFIG['batch_size']}", flush=True)
    print(f"Data directories: {DATA_DIRS}", flush=True)
    print(f"Models to run: {list(EXPERIMENTS.keys())}", flush=True)
    print(f"Runs per model: {CONFIG['num_runs']}", flush=True)
    print(f"Epochs per run: {CONFIG['epochs']}", flush=True)
    print("="*70, flush=True)

    # Enable cudnn benchmarking for faster convolutions
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster matrix ops
        torch.backends.cudnn.allow_tf32 = True

    all_results = {}

    for name, config in EXPERIMENTS.items():
        result = run_experiment_with_runs(name, config, num_runs=CONFIG['num_runs'])
        all_results[name] = result

        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(OUTPUT_DIR / f"intermediate_{timestamp}.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Print final table
    print("\n" + "="*80, flush=True)
    print("FINAL RESULTS - Main Comparison (Table 1)", flush=True)
    print("="*80, flush=True)
    print(f"{'Method':<20} {'Accuracy':<18} {'F1 Score':<18} {'AUC':<18}", flush=True)
    print("-"*80, flush=True)

    for name, result in all_results.items():
        s = result['summary']
        print(f"{name:<20} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}    "
              f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}    "
              f"{s['auc_mean']:.4f}±{s['auc_std']:.4f}", flush=True)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = OUTPUT_DIR / f"full_results_{timestamp}.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {final_file}", flush=True)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("="*70, flush=True)

    return all_results


if __name__ == '__main__':
    main()
