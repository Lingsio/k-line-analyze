"""
V2 Experiments - OHLC + GAF + 10-day/256x256
Based on:
1. Xiu et al. (2021): "(Re-)Imag(in)ing Price Trends" - OHLC bar charts
2. Chen & Tsai (2020): "Encoding candlesticks as images" - GAF encoding

Usage:
    python scripts/run_v2_experiments.py --experiment ohlc
    python scripts/run_v2_experiments.py --experiment gaf
    python scripts/run_v2_experiments.py --experiment all
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
import argparse

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIRS = [str(PROJECT_ROOT / 'data' / 'raw' / 'us')]
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'v2_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware-optimized config
NUM_WORKERS = min(8, multiprocessing.cpu_count())
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()
USE_COMPILE = hasattr(torch, 'compile')
PREFETCH_FACTOR = 4

# V2 Experiments - Key improvements
V2_EXPERIMENTS = {
    'OHLC-10d-256': {
        'description': 'OHLC bar charts (Xiu et al. 2021) - 10 days, 256x256',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'ohlc',
        'batch_size': 64,  # Smaller due to larger images
        'epochs': 30,
        'lr': 3e-4,
        'augment_prob': 0.3,
    },
    'GAF-10d-256': {
        'description': 'GAF encoding (Chen & Tsai 2020) - 10 days, 256x256',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'gaf',
        'gaf_method': 'gasf',
        'batch_size': 64,
        'epochs': 30,
        'lr': 3e-4,
        'augment_prob': 0.2,  # Less augmentation for GAF
    },
    'OHLC-GAF-10d-256': {
        'description': 'OHLC + GAF hybrid - 10 days, 256x256',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'ohlc',
        'use_gaf': True,
        'gaf_method': 'gasf',
        'batch_size': 64,
        'epochs': 30,
        'lr': 3e-4,
        'augment_prob': 0.3,
    },
    'Hybrid-10d-256': {
        'description': 'Hybrid OHLC/GAF blend - 10 days, 256x256',
        'window_size': 10,
        'img_size': (256, 256),
        'chart_type': 'hybrid',
        'batch_size': 64,
        'epochs': 30,
        'lr': 3e-4,
        'augment_prob': 0.3,
    },
    # Baseline for comparison
    'Candle-20d-128': {
        'description': 'Original candlestick - 20 days, 128x128 (baseline)',
        'window_size': 20,
        'img_size': (128, 128),
        'chart_type': 'candle',
        'batch_size': 128,
        'epochs': 20,
        'lr': 4e-4,
        'augment_prob': 0.3,
    },
}

# ============================================================================
# Training Functions
# ============================================================================

def create_dataloaders(config):
    """Create train/val/test dataloaders for V2 experiments."""
    common_args = {
        'data_dir': DATA_DIRS,
        'prediction_horizon': 5,
        'window_size': config['window_size'],
        'img_size': config['img_size'],
        'norm_method': 'robust',
        'label_threshold': 'dynamic',
        'chart_type': config.get('chart_type', 'candle'),
        'use_gaf': config.get('use_gaf', False),
        'gaf_method': config.get('gaf_method', 'gasf'),
    }

    train_ds = StockDataset(
        mode='train',
        augment_prob=config.get('augment_prob', 0.3),
        **common_args
    )
    val_ds = StockDataset(mode='val', augment_prob=0.0, **common_args)
    test_ds = StockDataset(mode='test', augment_prob=0.0, **common_args)

    loader_kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY,
        'prefetch_factor': PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        'persistent_workers': NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def create_model(num_classes, input_channels, img_size):
    """Create CNN model."""
    # ResNet18 handles various input sizes well
    model = CNNModel(
        num_classes=num_classes,
        input_channels=input_channels,
        arch='resnet18',
        pretrained=True
    )
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    use_amp = scaler is not None and DEVICE.type == 'cuda'

    for batch in loader:
        imgs, seqs, labels = batch
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=use_amp):
            inputs = imgs.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if torch.isnan(loss):
            continue

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


def evaluate(model, loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    use_amp = USE_AMP and DEVICE.type == 'cuda'

    with torch.no_grad():
        for batch in loader:
            imgs, seqs, labels = batch
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type='cuda', enabled=use_amp):
                inputs = imgs.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if not torch.isnan(loss):
                total_loss += loss.item()

            probs = torch.softmax(outputs.float(), dim=1)
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

    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"Description: {config['description']}")
    print(f"Config: window={config['window_size']}, img={config['img_size']}, "
          f"chart={config.get('chart_type', 'candle')}")
    print(f"{'='*70}")

    # Create data
    print(f"[seed={seed}] Loading data...")
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"[seed={seed}] Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training data!")
        return None

    # Create model
    input_channels = train_ds.get_num_channels()
    model = create_model(num_classes=2, input_channels=input_channels, img_size=config['img_size'])
    print(f"[seed={seed}] Model input channels: {input_channels}")

    # Compile model for faster execution
    if USE_COMPILE and DEVICE.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print(f"[seed={seed}] Model compiled with torch.compile")
        except Exception as e:
            print(f"[seed={seed}] torch.compile failed: {e}")

    # Training setup
    class_weights = train_ds.get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler() if USE_AMP and DEVICE.type == 'cuda' else None

    # Training loop
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    patience = 7
    history = []

    print(f"[seed={seed}] Training for up to {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
        })

        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_metrics = evaluate(model, test_loader, criterion)

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC:      {test_metrics['auc']:.4f}")

    return {
        'seed': seed,
        'config': config,
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'epochs_trained': len(history),
        'history': history,
    }


def run_experiment_with_runs(name, config, num_runs=3):
    """Run experiment multiple times."""
    print(f"\n{'#'*80}")
    print(f"# Running: {name}")
    print(f"# {config['description']}")
    print(f"{'#'*80}")

    all_runs = []
    for run_id in range(num_runs):
        seed = 42 + run_id * 100
        print(f"\n--- Run {run_id+1}/{num_runs} (seed={seed}) ---")
        result = run_single_experiment(name, config, seed=seed)
        if result:
            all_runs.append(result)

    if not all_runs:
        return None

    # Aggregate results
    metrics_keys = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    summary = {}
    for key in metrics_keys:
        values = [r['test_metrics'][key] for r in all_runs]
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))

    print(f"\n{'='*70}")
    print(f"Summary for {name}:")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  F1 Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  AUC:      {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"{'='*70}")

    return {
        'name': name,
        'config': config,
        'runs': all_runs,
        'summary': summary,
    }


def main():
    parser = argparse.ArgumentParser(description='V2 Experiments - OHLC + GAF')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=list(V2_EXPERIMENTS.keys()) + ['all'],
                       help='Which experiment to run')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs per experiment')
    args = parser.parse_args()

    print("="*80)
    print("V2 Experiments - OHLC Bar Charts + GAF Encoding")
    print("Based on: Xiu et al. (2021) and Chen & Tsai (2020)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"Mixed Precision: {USE_AMP}")
    print(f"Model Compilation: {USE_COMPILE}")
    print("="*80)

    # Enable cudnn benchmarking
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Select experiments
    if args.experiment == 'all':
        experiments_to_run = V2_EXPERIMENTS
    else:
        experiments_to_run = {args.experiment: V2_EXPERIMENTS[args.experiment]}

    all_results = {}

    for name, config in experiments_to_run.items():
        result = run_experiment_with_runs(name, config, num_runs=args.num_runs)
        if result:
            all_results[name] = result

            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(OUTPUT_DIR / f"v2_intermediate_{timestamp}.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # Print final comparison table
    print("\n" + "="*90)
    print("FINAL RESULTS - V2 Experiments Comparison")
    print("="*90)
    print(f"{'Method':<20} {'Description':<40} {'Accuracy':<18} {'F1 Score':<18}")
    print("-"*90)

    for name, result in all_results.items():
        s = result['summary']
        desc = result['config']['description'][:38]
        print(f"{name:<20} {desc:<40} "
              f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}  "
              f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = OUTPUT_DIR / f"v2_final_results_{timestamp}.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {final_file}")
    print("="*90)


if __name__ == '__main__':
    main()
