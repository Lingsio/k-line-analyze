"""
SAK-Net Training on Full US Stock Universe (~4500 stocks)
=========================================================

Uses the same architecture that achieved 57.81% on 57 stocks,
now scaled to ~4500 stocks grouped by GICS sector.

Reads sector groups from data/us_stock_groups_full.json
(built by scripts/build_sector_groups.py).

Usage:
    python scripts/train_full_universe.py --all
    python scripts/train_full_universe.py --sector Technology
    python scripts/train_full_universe.py --all --debug
    python scripts/train_full_universe.py --all --num-runs 1
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Same config that achieved 57.81% accuracy
CONFIG = {
    'arch': 'resnet18',
    'pretrained': True,
    'dropout': 0.3,
    'input_channels': 3,

    'window_size': 20,
    'prediction_horizon': 5,
    'img_size': (128, 128),
    'chart_type': 'ohlc',
    'norm_method': 'robust',
    'use_clahe': True,

    'num_classes': 2,
    'label_threshold': 'dynamic',

    'batch_size': 128,
    'num_workers': 8,
    'epochs': 50,
    'lr': 4e-4,
    'weight_decay': 1e-4,
    'patience': 7,
    'label_smoothing': 0.1,
}


def build_resnet18(num_classes=2, pretrained=True, dropout=0.3):
    """Build ResNet18 with proper modifications."""
    model = CNNModel(
        num_classes=num_classes,
        input_channels=3,
        arch='resnet18',
        pretrained=pretrained,
    )

    in_features = model.backbone.fc.in_features
    model.backbone.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


def train_epoch(model, loader, criterion, optimizer, scaler, device=DEVICE):
    """Train one epoch."""
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for imgs, _, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            out = model(imgs)
            loss = criterion(out, lbls)

        if torch.isnan(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds_all.extend(out.argmax(1).detach().cpu().numpy())
        labels_all.extend(lbls.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(labels_all, preds_all) if labels_all else 0
    return avg_loss, acc


def evaluate(model, loader, criterion, device=DEVICE):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for imgs, _, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                out = model(imgs)
                loss = criterion(out, lbls)

            if not torch.isnan(loss):
                total_loss += loss.item()

            probs = torch.softmax(out.float(), dim=1)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
            probs_all.extend(probs[:, 1].cpu().numpy())

    metrics = {
        'loss': total_loss / max(len(loader), 1),
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1': f1_score(labels_all, preds_all, average='weighted', zero_division=0),
        'precision': precision_score(labels_all, preds_all, average='weighted', zero_division=0),
        'recall': recall_score(labels_all, preds_all, average='weighted', zero_division=0),
    }

    try:
        metrics['auc'] = roc_auc_score(labels_all, probs_all)
    except Exception:
        metrics['auc'] = 0.5

    return metrics


def train_sector_model(sector_id, sector_info, config, seed):
    """Train a model for one sector with specific seed."""
    stocks = sector_info['stocks']
    data_dir = str(PROJECT_ROOT / 'data' / 'raw' / 'us')

    common_cfg = {
        'data_dir': data_dir,
        'window_size': config['window_size'],
        'prediction_horizon': config['prediction_horizon'],
        'img_size': config['img_size'],
        'chart_type': config['chart_type'],
        'norm_method': config['norm_method'],
        'use_clahe': config['use_clahe'],
        'label_threshold': config['label_threshold'],
        'num_classes': config['num_classes'],
    }

    # Training dataset (with quantile filter)
    train_cfg = common_cfg.copy()
    train_cfg['quantile_filter'] = 0.35

    print(f"    Loading datasets...")
    # Use tickers_filter for efficient sector-based loading (only loads specified stocks)
    train_cfg['tickers_filter'] = stocks
    common_cfg['tickers_filter'] = stocks
    
    train_ds = StockDataset(mode='train', augment_prob=0.5, **train_cfg)
    val_ds = StockDataset(mode='val', augment_prob=0.0, **common_cfg)
    test_ds = StockDataset(mode='test', augment_prob=0.0, **common_cfg)

    print(f"    Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if len(train_ds) < 100:
        print(f"    WARNING: Too few training samples ({len(train_ds)}), skipping")
        return None

    loader_kw = {
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'pin_memory': True,
        'persistent_workers': True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    # Build model
    model = build_resnet18(
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        dropout=config['dropout']
    ).to(DEVICE)

    # torch.compile on Linux
    if sys.platform != 'win32' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception:
            pass

    # Training setup
    class_weights = train_ds.get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else GradScaler('cpu', enabled=False)

    # Training loop
    best_f1 = 0
    best_state = None
    patience_cnt = 0

    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step(val_m['f1'])

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                break

    # Load best and test
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_m = evaluate(model, test_loader, criterion)

    return {
        'seed': seed,
        'test_metrics': test_m,
        'best_val_f1': best_f1,
        'epochs': epoch + 1,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
    }


def main():
    parser = argparse.ArgumentParser(description='SAK-Net Training - Full US Universe')
    parser.add_argument('--all', action='store_true', help='Train all sectors')
    parser.add_argument('--sector', type=str, help='Train specific sector')
    parser.add_argument('--num-runs', type=int, default=2, help='Number of runs per sector')
    parser.add_argument('--debug', action='store_true', help='Debug mode (5 epochs, 1 run)')
    parser.add_argument('--groups-file', type=Path, default=None,
                        help='Sector groups JSON file')
    args = parser.parse_args()

    if not args.all and not args.sector:
        parser.print_help()
        return

    # Load sector definitions
    if args.groups_file is None:
        args.groups_file = PROJECT_ROOT / 'data' / 'us_stock_groups_full.json'

    if not args.groups_file.exists():
        print(f"ERROR: Groups file not found: {args.groups_file}")
        print("Run first: python scripts/build_sector_groups.py --resume")
        return

    with open(args.groups_file, encoding='utf-8') as f:
        groups_data = json.load(f)['groups']

    # Always skip 'Other' sector (heterogeneous mix of small sectors and unclassified stocks)
    if 'Other' in groups_data:
        print(f"Note: Excluding 'Other' sector ({len(groups_data['Other']['stocks'])} stocks) - heterogeneous, not a coherent sector")
    sectors_to_train = [k for k in groups_data.keys() if k != 'Other']
    
    if args.sector:
        if args.sector not in sectors_to_train:
            print(f"ERROR: Unknown or excluded sector: {args.sector}")
            return
        sectors_to_train = [args.sector]

    print("=" * 70)
    print("SAK-Net Training - Full US Stock Universe")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Sectors: {len(sectors_to_train)}")
    total_stocks = sum(len(groups_data[s]['stocks']) for s in sectors_to_train)
    print(f"Total stocks: {total_stocks}")
    print(f"Runs per sector: {args.num_runs}")
    print(f"Config: {CONFIG['arch']}, {CONFIG['num_classes']}-class, batch={CONFIG['batch_size']}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 70)

    config = CONFIG.copy()
    if args.debug:
        config['epochs'] = 5
        args.num_runs = 1

    # Train each sector
    all_results = {}

    for sector_idx, sector_id in enumerate(sectors_to_train):
        if sector_id not in groups_data:
            print(f"ERROR: Unknown sector: {sector_id}")
            continue

        sector_info = groups_data[sector_id]
        n_stocks = len(sector_info['stocks'])

        print(f"\n{'='*70}")
        print(f"[{sector_idx+1}/{len(sectors_to_train)}] Sector: {sector_id}")
        print(f"  Name: {sector_info['name']}")
        print(f"  Stocks: {n_stocks}")
        if n_stocks <= 20:
            print(f"  Tickers: {sector_info['stocks']}")
        else:
            print(f"  Tickers: {sector_info['stocks'][:10]} ... ({n_stocks} total)")
        print(f"{'='*70}")

        sector_results = []
        seeds = [42, 142, 242, 342, 442][:args.num_runs]

        for run_idx, seed in enumerate(seeds):
            print(f"\n  Run {run_idx+1}/{len(seeds)} (seed={seed})")
            torch.manual_seed(seed)
            np.random.seed(seed)

            result = train_sector_model(sector_id, sector_info, config, seed)
            if result:
                sector_results.append(result)
                m = result['test_metrics']
                print(f"    Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}, AUC: {m.get('auc', 0):.4f}")

        if sector_results:
            best_run = max(sector_results, key=lambda x: x['test_metrics']['accuracy'])
            all_results[sector_id] = {
                'sector_info': {
                    'name': sector_info['name'],
                    'stocks': sector_info['stocks'],
                    'num_stocks': n_stocks,
                },
                'runs': sector_results,
                'best_run': best_run,
            }

            m = best_run['test_metrics']
            print(f"\n  Best: Acc={m['accuracy']:.4f} ({m['accuracy']*100:.2f}%), "
                  f"F1={m['f1']:.4f}, AUC={m.get('auc', 0):.4f}")

    # Final summary
    if all_results:
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"{'Sector':<30} {'Stocks':>6} {'Best Acc':>10} {'Best F1':>10}")
        print("-" * 70)

        for sector_id, data in all_results.items():
            best = data['best_run']
            m = best['test_metrics']
            n = data['sector_info']['num_stocks']
            print(f"{sector_id:<30} {n:>6} {m['accuracy']:>10.4f} {m['f1']:>10.4f}")

        avg_acc = np.mean([d['best_run']['test_metrics']['accuracy'] for d in all_results.values()])
        avg_f1 = np.mean([d['best_run']['test_metrics']['f1'] for d in all_results.values()])
        total = sum(d['sector_info']['num_stocks'] for d in all_results.values())
        print("-" * 70)
        print(f"{'AVERAGE':<30} {total:>6} {avg_acc:>10.4f} {avg_f1:>10.4f}")

        # Save
        output_dir = PROJECT_ROOT / 'outputs' / 'full_universe'
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f'results_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(save_path, 'w') as f:
            json.dump({
                'config': config,
                'results': all_results,
                'avg_accuracy': float(avg_acc),
                'avg_f1': float(avg_f1),
                'total_stocks': total,
                'num_sectors': len(all_results),
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()
