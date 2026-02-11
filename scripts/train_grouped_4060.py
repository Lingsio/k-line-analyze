"""
Grouped Training for RTX 4060 (8GB VRAM)
=========================================

Train sector-specific models using lightweight CNN.
Optimized for RTX 4060 8GB VRAM.

Features:
- Lightweight CNN (~500K params vs ResNet18's 11M)
- Batch size 32 (fits 8GB VRAM comfortably)
- 3-class classification (Up/Neutral/Down)
- Per-sector training with data leakage prevention

Usage:
    # Train all sectors
    python scripts/train_grouped_4060.py --all
    
    # Train specific sector
    python scripts/train_grouped_4060.py --sector Tech_Software
    
    # 2-class mode (no neutral)
    python scripts/train_grouped_4060.py --all --num-classes 2
    
    # Debug mode (faster)
    python scripts/train_grouped_4060.py --all --debug
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.lightweight_cnn import build_lightweight_cnn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RTX 4060 Optimized Config
RTX4060_CONFIG = {
    'batch_size': 32,           # Fits 8GB VRAM comfortably
    'num_workers': 4,           # Adjust based on your CPU
    'window_size': 20,          # 20 days
    'prediction_horizon': 5,    # Predict 5 days ahead
    'img_size': (128, 128),     # Input image size
    'label_threshold': 0.01,    # 1% threshold for neutral
    'num_classes': 3,           # 3-class by default
    'dropout': 0.3,
    'cnn_variant': 'light',     # 'light' or 'ultra'
    'epochs': 50,
    'lr': 1e-3,                 # Slightly higher LR for small model
    'weight_decay': 1e-4,
    'patience': 10,
    'label_smoothing': 0.1,
}


def train_epoch(model, loader, criterion, optimizer, scaler, device=DEVICE):
    """Train one epoch with mixed precision."""
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
    preds_all, labels_all = [], []
    
    with torch.no_grad():
        for imgs, _, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                out = model(imgs)
                loss = criterion(out, lbls)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
    
    metrics = {
        'loss': total_loss / max(len(loader), 1),
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1': f1_score(labels_all, preds_all, average='weighted', zero_division=0),
        'precision': precision_score(labels_all, preds_all, average='weighted', zero_division=0),
        'recall': recall_score(labels_all, preds_all, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    if len(set(labels_all)) <= 3:
        for cls_idx, cls_name in [(0, 'down'), (1, 'neutral'), (2, 'up')]:
            if cls_idx in labels_all:
                metrics[f'recall_{cls_name}'] = recall_score(
                    np.array(labels_all) == cls_idx,
                    np.array(preds_all) == cls_idx,
                    zero_division=0
                )
    
    return metrics


def train_sector_model(sector_id, sector_info, config, args):
    """Train a model for one sector."""
    stocks = sector_info['stocks']
    print(f"\n{'='*70}")
    print(f"Training Sector: {sector_id} ({sector_info['name']})")
    print(f"Stocks: {stocks}")
    print(f"{'='*70}")
    
    # Create datasets
    data_dir = str(PROJECT_ROOT / 'data' / 'raw' / 'us')
    
    # Common config (NO filtering for val/test to avoid leakage)
    common_cfg = {
        'data_dir': data_dir,
        'window_size': config['window_size'],
        'prediction_horizon': config['prediction_horizon'],
        'img_size': config['img_size'],
        'label_threshold': config['label_threshold'],
        'num_classes': config['num_classes'],
        'chart_type': 'candle',
        'norm_method': 'robust',
        'use_clahe': True,
    }
    
    limit = 2 if args.debug else None
    
    # Training dataset (with optional filter for 3-class)
    train_cfg = common_cfg.copy()
    if config['num_classes'] == 3:
        train_cfg['train_filter_threshold'] = 0.003  # Filter extreme neutrals
    
    print(f"\nLoading datasets...")
    train_ds_full = StockDataset(mode='train', augment_prob=0.5, limit=limit, **train_cfg)
    val_ds_full = StockDataset(mode='val', augment_prob=0.0, limit=limit, **common_cfg)
    test_ds_full = StockDataset(mode='test', augment_prob=0.0, limit=limit, **common_cfg)
    
    # Filter by sector stocks
    stock_set = set(stocks)
    train_idx = [i for i, s in enumerate(train_ds_full.samples) if s['ticker'] in stock_set]
    val_idx = [i for i, s in enumerate(val_ds_full.samples) if s['ticker'] in stock_set]
    test_idx = [i for i, s in enumerate(test_ds_full.samples) if s['ticker'] in stock_set]
    
    print(f"Samples: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    if len(train_idx) < 50:
        print(f"WARNING: Too few training samples ({len(train_idx)}), skipping this sector")
        return None
    
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)
    test_ds = Subset(test_ds_full, test_idx)
    
    # Check class distribution
    train_labels = [train_ds_full.samples[i]['label'] for i in train_idx]
    label_counts = {i: train_labels.count(i) for i in set(train_labels)}
    print(f"Train class distribution: {label_counts}")
    
    # DataLoaders
    loader_kw = {
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'pin_memory': True,
        'persistent_workers': config['num_workers'] > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    
    # Build lightweight model
    print(f"\nBuilding {config['cnn_variant']} CNN...")
    model = build_lightweight_cnn(
        variant=config['cnn_variant'],
        num_classes=config['num_classes'],
        input_channels=3,
        dropout=config['dropout']
    ).to(DEVICE)
    
    # Print VRAM usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
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
    history = []
    
    print(f"\nTraining for up to {config['epochs']} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>10} {'Train Acc':>10} {'Val Acc':>10} {'Val F1':>10} {'LR':>12}")
    print("-" * 70)
    
    epochs = 5 if args.debug else config['epochs']
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_m['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_m['accuracy'],
            'val_f1': val_m['f1'],
            'lr': current_lr,
        })
        
        if (epoch + 1) % 5 == 0 or epoch < 3:
            print(f"{epoch+1:>6} {train_loss:>10.4f} {train_acc:>10.4f} "
                  f"{val_m['accuracy']:>10.4f} {val_m['f1']:>10.4f} {current_lr:>12.6f}")
        
        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best and evaluate on test set
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    test_m = evaluate(model, test_loader, criterion)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_m['accuracy']:.4f}")
    print(f"  F1:       {test_m['f1']:.4f}")
    print(f"  Precision:{test_m['precision']:.4f}")
    print(f"  Recall:   {test_m['recall']:.4f}")
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak_vram:.2f} GB")
    
    # Save model
    output_dir = PROJECT_ROOT / 'outputs' / 'grouped_models_4060'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f'{sector_id}_{config["num_classes"]}class.pt'
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'config': config,
        'sector_info': sector_info,
        'test_metrics': test_m,
        'history': history,
        'best_val_f1': best_f1,
    }, save_path)
    print(f"Model saved to: {save_path}")
    
    return {
        'sector_id': sector_id,
        'sector_name': sector_info['name'],
        'num_stocks': len(stocks),
        'test_metrics': test_m,
        'best_val_f1': best_f1,
    }


def main():
    parser = argparse.ArgumentParser(description='Grouped Training for RTX 4060')
    parser.add_argument('--all', action='store_true', help='Train all sectors')
    parser.add_argument('--sector', type=str, help='Train specific sector (e.g., Tech_Software)')
    parser.add_argument('--num-classes', type=int, default=3, choices=[2, 3], help='Number of classes')
    parser.add_argument('--debug', action='store_true', help='Debug mode (faster)')
    parser.add_argument('--variant', type=str, default='light', choices=['light', 'ultra'], 
                        help='CNN variant: light (~500K params) or ultra (~100K params)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32 for 8GB)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()
    
    if not args.all and not args.sector:
        parser.print_help()
        print("\nError: Must specify --all or --sector")
        return
    
    # Load sector definitions
    group_file = PROJECT_ROOT / 'data' / 'us_stock_groups.json'
    with open(group_file, encoding='utf-8') as f:
        groups_data = json.load(f)['groups']
    
    # Determine which sectors to train
    if args.all:
        sectors_to_train = list(groups_data.keys())
    else:
        sectors_to_train = [args.sector]
    
    print("=" * 70)
    print("Grouped Training for RTX 4060 (8GB VRAM)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Sectors: {sectors_to_train}")
    print(f"Classes: {args.num_classes}")
    print(f"CNN Variant: {args.variant}")
    print(f"Batch Size: {args.batch_size}")
    print()
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Update config
    config = RTX4060_CONFIG.copy()
    config['num_classes'] = args.num_classes
    config['cnn_variant'] = args.variant
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    
    # Train each sector
    results = []
    for sector_id in sectors_to_train:
        if sector_id not in groups_data:
            print(f"ERROR: Unknown sector: {sector_id}")
            continue
        
        result = train_sector_model(sector_id, groups_data[sector_id], config, args)
        if result:
            results.append(result)
    
    # Summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY: All Sectors")
        print(f"{'='*70}")
        print(f"{'Sector':<25} {'Stocks':>6} {'Test Acc':>10} {'Test F1':>10}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['sector_id']:<25} {r['num_stocks']:>6} "
                  f"{r['test_metrics']['accuracy']:>10.4f} {r['test_metrics']['f1']:>10.4f}")
        
        # Average
        avg_acc = np.mean([r['test_metrics']['accuracy'] for r in results])
        avg_f1 = np.mean([r['test_metrics']['f1'] for r in results])
        print("-" * 70)
        print(f"{'AVERAGE':<25} {'':>6} {avg_acc:>10.4f} {avg_f1:>10.4f}")
        
        # Save summary
        summary_path = PROJECT_ROOT / 'outputs' / 'grouped_models_4060' / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'config': config,
                'results': results,
                'avg_accuracy': float(avg_acc),
                'avg_f1': float(avg_f1),
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
