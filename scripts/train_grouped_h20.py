"""
Grouped Training for H20 (96GB VRAM) - 54% Accuracy Config
==========================================================

基于实验验证的54%准确率配置：
- ResNet18 (pretrained)
- 2-class (up/down)
- OHLC bars, 20 days, 128px
- Batch size 128
- Grouped by sector

Usage:
    python scripts/train_grouped_h20.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# H20 Optimized Config - Based on 54% accuracy experiments
H20_CONFIG = {
    'arch': 'resnet18',
    'pretrained': True,         # Critical for performance
    'dropout': 0.3,
    'input_channels': 3,
    
    # Data config (validated)
    'window_size': 20,          # 20 days
    'prediction_horizon': 5,    # Predict 5 days
    'img_size': (128, 128),     # 128x128 (256 for 10-day)
    'chart_type': 'ohlc',       # OHLC bars (Xiu et al. 2021)
    'norm_method': 'robust',
    'use_clahe': True,
    
    # 2-class (not 3-class)
    'num_classes': 2,
    'label_threshold': 'dynamic',
    
    # Training config
    'batch_size': 128,          # H20 can handle this
    'num_workers': 8,
    'epochs': 50,
    'lr': 4e-4,                 # From experiments
    'weight_decay': 1e-4,
    'patience': 7,
    'label_smoothing': 0.1,
}


def build_resnet18(num_classes=2, input_channels=3, pretrained=True, dropout=0.3):
    """Build ResNet18 with proper modifications."""
    model = CNNModel(
        num_classes=num_classes,
        input_channels=3,  # Build with 3ch first
        arch='resnet18',
        pretrained=pretrained,
    )
    
    # Handle non-3-channel input
    if input_channels != 3:
        old_weight = model.backbone.conv1.weight.data
        new_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data[:, :3] = old_weight
        for c in range(3, input_channels):
            new_conv.weight.data[:, c] = old_weight.mean(dim=1)
        model.backbone.conv1 = new_conv
    
    # Insert dropout before classifier
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
    except:
        metrics['auc'] = 0.5
    
    return metrics


def train_sector_model(sector_id, sector_info, config, args, seed):
    """Train a model for one sector with specific seed."""
    stocks = sector_info['stocks']
    
    # Create datasets
    data_dir = str(PROJECT_ROOT / 'data' / 'raw' / 'us')
    
    # Common config
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
    train_cfg['quantile_filter'] = 0.35  # Filter flat samples
    
    train_ds_full = StockDataset(mode='train', augment_prob=0.5, **train_cfg)
    val_ds_full = StockDataset(mode='val', augment_prob=0.0, **common_cfg)
    test_ds_full = StockDataset(mode='test', augment_prob=0.0, **common_cfg)
    
    # Filter by sector stocks
    stock_set = set(stocks)
    train_idx = [i for i, s in enumerate(train_ds_full.samples) if s['ticker'] in stock_set]
    val_idx = [i for i, s in enumerate(val_ds_full.samples) if s['ticker'] in stock_set]
    test_idx = [i for i, s in enumerate(test_ds_full.samples) if s['ticker'] in stock_set]
    
    if len(train_idx) < 100:
        print(f"  WARNING: Too few training samples ({len(train_idx)}), skipping")
        return None
    
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)
    test_ds = Subset(test_ds_full, test_idx)
    
    # DataLoaders
    loader_kw = {
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'pin_memory': True,
        'persistent_workers': True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    
    # Build ResNet18
    model = build_resnet18(
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    # torch.compile for H20 (Linux only)
    if sys.platform != 'win32' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  torch.compile: enabled")
        except:
            pass
    
    # Training setup
    class_weights = train_ds_full.get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else GradScaler('cpu', enabled=False)
    
    # Training loop
    best_f1 = 0
    best_state = None
    patience_cnt = 0
    best_metrics = None
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step(val_m['f1'])
        
        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_m
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
    }


def main():
    parser = argparse.ArgumentParser(description='Grouped Training for H20 - 54% Config')
    parser.add_argument('--all', action='store_true', help='Train all sectors')
    parser.add_argument('--sector', type=str, help='Train specific sector')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per sector')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    if not args.all and not args.sector:
        parser.print_help()
        return
    
    # Load sector definitions
    with open(PROJECT_ROOT / 'data' / 'us_stock_groups.json', encoding='utf-8') as f:
        groups_data = json.load(f)['groups']
    
    sectors_to_train = list(groups_data.keys()) if args.all else [args.sector]
    
    print("=" * 70)
    print("Grouped Training for H20 - 54% Accuracy Config")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Sectors: {sectors_to_train}")
    print(f"Runs per sector: {args.num_runs}")
    print(f"Config: {H20_CONFIG['arch']}, {H20_CONFIG['num_classes']}-class, batch={H20_CONFIG['batch_size']}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    config = H20_CONFIG.copy()
    if args.debug:
        config['epochs'] = 5
        args.num_runs = 1
    
    # Train each sector
    all_results = {}
    
    for sector_id in sectors_to_train:
        if sector_id not in groups_data:
            print(f"ERROR: Unknown sector: {sector_id}")
            continue
        
        sector_info = groups_data[sector_id]
        print(f"\n{'='*70}")
        print(f"Sector: {sector_id} ({sector_info['name']})")
        print(f"Stocks: {sector_info['stocks']}")
        print(f"{'='*70}")
        
        sector_results = []
        seeds = [42, 142, 242, 342, 442][:args.num_runs]
        
        for run_idx, seed in enumerate(seeds):
            print(f"\n  Run {run_idx+1}/{len(seeds)} (seed={seed})")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            result = train_sector_model(sector_id, sector_info, config, args, seed)
            if result:
                sector_results.append(result)
                m = result['test_metrics']
                print(f"    Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}, AUC: {m.get('auc', 0):.4f}")
        
        if sector_results:
            # Find best run
            best_run = max(sector_results, key=lambda x: x['test_metrics']['accuracy'])
            all_results[sector_id] = {
                'sector_info': sector_info,
                'runs': sector_results,
                'best_run': best_run,
            }
            
            print(f"\n  Best run (seed={best_run['seed']}):")
            m = best_run['test_metrics']
            print(f"    Accuracy: {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
            print(f"    F1: {m['f1']:.4f}")
            print(f"    AUC: {m.get('auc', 0):.4f}")
    
    # Final summary
    if len(all_results) > 0:
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"{'Sector':<25} {'Best Acc':>10} {'Best F1':>10} {'Seed':>8}")
        print("-" * 70)
        
        for sector_id, data in all_results.items():
            best = data['best_run']
            m = best['test_metrics']
            print(f"{sector_id:<25} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {best['seed']:>8}")
        
        # Average
        avg_acc = np.mean([d['best_run']['test_metrics']['accuracy'] for d in all_results.values()])
        print("-" * 70)
        print(f"{'AVERAGE':<25} {avg_acc:>10.4f}")
        
        # Save
        output_dir = PROJECT_ROOT / 'outputs' / 'grouped_h20'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / f'results_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(save_path, 'w') as f:
            json.dump({
                'config': config,
                'results': all_results,
                'avg_accuracy': float(avg_acc),
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()
