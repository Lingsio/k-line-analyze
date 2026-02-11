"""
3-Class Stock Prediction (Up/Neutral/Down)
===========================================

This script trains a 3-class CNN predictor with special evaluation:
- Label 0: Down (return < -threshold)
- Label 1: Neutral (|return| <= threshold)  
- Label 2: Up (return > threshold)

Special evaluation rule:
- Predicting "neutral" but actual move exceeds threshold = WRONG
- This prevents the model from being "lazy" and always predicting neutral

Data Leakage Prevention:
- quantile_filter and train_filter_threshold ONLY apply to training set
- Validation and test sets keep ALL samples for unbiased evaluation

Usage:
    python scripts/train_3class_predictor.py --threshold 0.01 --epochs 40
    python scripts/train_3class_predictor.py --debug
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(arch='resnet18', input_channels=3, num_classes=3, pretrained=True, dropout=0.3):
    """Build CNN model for 3-class classification."""
    model = CNNModel(num_classes=num_classes, input_channels=3, arch=arch, pretrained=pretrained)
    
    if input_channels != 3 and arch == 'resnet18':
        old_weight = model.backbone.conv1.weight.data
        new_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data[:, :3] = old_weight
        for c in range(3, input_channels):
            new_conv.weight.data[:, c] = old_weight.mean(dim=1)
        model.backbone.conv1 = new_conv
    
    if arch == 'resnet18':
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
        with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
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


def evaluate_model(model, loader, criterion, device=DEVICE):
    """Standard evaluation."""
    model.eval()
    total_loss = 0
    preds_all, labels_all, probs_all = [], [], []
    
    with torch.no_grad():
        for imgs, _, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
                out = model(imgs)
                loss = criterion(out, lbls)
            if not torch.isnan(loss):
                total_loss += loss.item()
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
            probs_all.extend(torch.softmax(out.float(), dim=1).cpu().numpy())
    
    metrics = {
        'loss': total_loss / max(len(loader), 1),
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1': f1_score(labels_all, preds_all, average='weighted', zero_division=0),
        'precision': precision_score(labels_all, preds_all, average='weighted', zero_division=0),
        'recall': recall_score(labels_all, preds_all, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    for cls_idx, cls_name in [(0, 'down'), (1, 'neutral'), (2, 'up')]:
        cls_mask = np.array(labels_all) == cls_idx
        if cls_mask.sum() > 0:
            metrics[f'recall_{cls_name}'] = recall_score(
                np.array(labels_all) == cls_idx, 
                np.array(preds_all) == cls_idx, 
                zero_division=0
            )
    
    return metrics


def evaluate_with_neutral_penalty(model, loader, neutral_threshold=0.01, device=DEVICE):
    """
    Special 3-class evaluation:
    - Predicting neutral (1) but actual return exceeds threshold = WRONG
    """
    model.eval()
    preds_all, labels_all, raw_returns_all = [], [], []
    dataset = loader.dataset
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            imgs, _, lbls = batch
            imgs = imgs.to(device, non_blocking=True)
            
            with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
                out = model(imgs)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
            
            # Get raw returns
            batch_size = len(lbls)
            start_idx = batch_idx * loader.batch_size
            for i in range(batch_size):
                sample_idx = start_idx + i
                if sample_idx < len(dataset.samples):
                    raw_returns_all.append(dataset.samples[sample_idx].get('raw_return', 0))
                else:
                    raw_returns_all.append(0)
    
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    raw_returns_all = np.array(raw_returns_all)
    
    # Standard metrics
    metrics = {
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1_weighted': f1_score(labels_all, preds_all, average='weighted', zero_division=0),
    }
    
    # Strict accuracy: neutral predictions on non-neutral actuals are wrong
    strict_correct = 0
    for i in range(len(preds_all)):
        pred = preds_all[i]
        actual_label = labels_all[i]
        ret = raw_returns_all[i]
        actual_is_neutral = abs(ret) < neutral_threshold
        
        if pred == 1:  # Predicted neutral
            if actual_is_neutral:
                strict_correct += 1
        elif pred == actual_label:
            strict_correct += 1
    
    metrics['strict_accuracy'] = strict_correct / len(preds_all) if len(preds_all) > 0 else 0
    
    # Count predictions
    metrics['pred_down'] = int((preds_all == 0).sum())
    metrics['pred_neutral'] = int((preds_all == 1).sum())
    metrics['pred_up'] = int((preds_all == 2).sum())
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='3-Class Stock Predictor')
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data' / 'raw' / 'us'))
    parser.add_argument('--threshold', type=float, default=0.01, help='Neutral threshold (default: 0.01 = 1%)')
    parser.add_argument('--train-filter', type=float, default=0.003, help='Training filter threshold (default: 0.003)')
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    print("=" * 70)
    print("3-Class Stock Predictor (Up/Neutral/Down)")
    print("=" * 70)
    print(f"Neutral threshold: {args.threshold:.2%}")
    print(f"Training filter: {args.train_filter:.2%}")
    print(f"Device: {DEVICE}")
    
    # Config
    config = {
        'window_size': args.window_size,
        'prediction_horizon': args.horizon,
        'img_size': (128, 128),
        'label_threshold': args.threshold,
        'num_classes': 3,
        'chart_type': 'candle',
        'norm_method': 'robust',
        'use_clahe': True,
        'augment_prob': 0.5 if not args.debug else 0.0,
        'epochs': args.epochs if not args.debug else 5,
        'batch_size': args.batch_size,
        'lr': args.lr,
    }
    
    # Create datasets - IMPORTANT: only training set gets filter
    print("\nLoading datasets...")
    
    # Training set with filter
    train_ds = StockDataset(
        data_dir=args.data_dir,
        mode='train',
        limit=5 if args.debug else None,
        train_filter_threshold=args.train_filter,
        **{k: v for k, v in config.items() if k not in ['epochs', 'batch_size', 'lr', 'augment_prob']}
    )
    
    # Val/Test set WITHOUT filter (to avoid data leakage)
    common_cfg = {k: v for k, v in config.items() if k not in ['epochs', 'batch_size', 'lr', 'augment_prob', 'train_filter_threshold']}
    val_ds = StockDataset(data_dir=args.data_dir, mode='val', limit=5 if args.debug else None, **common_cfg)
    test_ds = StockDataset(data_dir=args.data_dir, mode='test', limit=5 if args.debug else None, **common_cfg)
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples")
    
    # Check class distribution
    for name, ds in [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]:
        labels = [s['label'] for s in ds.samples]
        print(f"{name} distribution: Down={labels.count(0)}, Neutral={labels.count(1)}, Up={labels.count(2)}")
    
    # DataLoaders
    loader_kw = {'batch_size': config['batch_size'], 'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    
    # Model
    model = build_model(num_classes=3, dropout=args.dropout).to(DEVICE)
    
    # Training
    class_weights = train_ds.get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else GradScaler('cpu', enabled=False)
    
    best_f1 = 0
    best_state = None
    patience = 10
    patience_cnt = 0
    
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config['epochs']}: loss={train_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_m['accuracy']:.4f}, val_f1={val_m['f1']:.4f}")
        
        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best and evaluate
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    # Standard evaluation
    test_m = evaluate_model(model, test_loader, criterion)
    print(f"\nStandard Metrics:")
    print(f"  Accuracy: {test_m['accuracy']:.4f}")
    print(f"  F1 (weighted): {test_m['f1']:.4f}")
    print(f"  Recall Down: {test_m.get('recall_down', 0):.4f}")
    print(f"  Recall Neutral: {test_m.get('recall_neutral', 0):.4f}")
    print(f"  Recall Up: {test_m.get('recall_up', 0):.4f}")
    
    # Strict evaluation with neutral penalty
    strict_m = evaluate_with_neutral_penalty(model, test_loader, neutral_threshold=args.threshold)
    print(f"\nStrict Metrics (neutral penalty):")
    print(f"  Strict Accuracy: {strict_m['strict_accuracy']:.4f}")
    print(f"  Predictions: Down={strict_m['pred_down']}, Neutral={strict_m['pred_neutral']}, Up={strict_m['pred_up']}")
    
    # Save
    output_dir = PROJECT_ROOT / 'outputs' / '3class_models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f'3class_model_{datetime.now():%Y%m%d_%H%M%S}.pt'
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'config': config,
        'test_metrics': test_m,
        'strict_metrics': strict_m,
    }, save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == '__main__':
    main()
