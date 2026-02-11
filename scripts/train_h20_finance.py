"""
H20-optimized training script for high-resolution financial CNN.

Features:
- Large batch sizes (up to 256 for standard, 32 for h20_max)
- Mixed precision training (AMP)
- Multi-GPU support (DataParallel)
- Gradient accumulation for very large models
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hr_dataset import HighResStockDataset
from src.data.sector_config import US_STOCK_GROUPS
from src.models.finance_cnn_h20 import FinanceCNN_H20, FinanceCNN_Large, create_model

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Enable TF32 for faster training on H20
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'h20_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# H20 optimized configs
H20_CONFIGS = {
    'standard': {
        'resolution': 'standard',
        'batch_size': 256,
        'base_channels': 64,
        'num_blocks': 3,
        'model_type': 'finance_h20',
    },
    'high': {
        'resolution': 'high',
        'batch_size': 128,
        'base_channels': 128,
        'num_blocks': 4,
        'model_type': 'finance_h20',
    },
    'ultra': {
        'resolution': 'ultra',
        'batch_size': 64,
        'base_channels': 128,
        'num_blocks': 5,
        'model_type': 'finance_h20',
    },
    'h20_max': {
        'resolution': 'h20_max',
        'batch_size': 32,
        'base_channels': 256,
        'num_blocks': 6,
        'model_type': 'finance_large',
    },
}


def train_epoch_amp(model, dataloader, criterion, optimizer, scaler, use_amp=True):
    """Train one epoch with mixed precision."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward with scaler
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
    }
    return metrics


def train_sector_model(sector, config, run_id=0):
    """Train a model for specific sector with H20 optimizations."""
    print(f"\n{'='*80}")
    print(f"Training: {sector} | Resolution: {config['resolution']} | Run {run_id+1}")
    print(f"{'='*80}")
    
    torch.manual_seed(42 + run_id)
    np.random.seed(42 + run_id)
    
    # Create datasets
    print("Loading data...")
    train_dataset = HighResStockDataset(
        data_dir=PROJECT_ROOT / 'data' / 'raw' / 'us',
        years=config['train_years'],
        window_size=config['window_size'],
        prediction_horizon=config['prediction_horizon'],
        sector=sector,
        resolution=config['resolution'],
        use_volume=True,
    )
    
    test_dataset = HighResStockDataset(
        data_dir=PROJECT_ROOT / 'data' / 'raw' / 'us',
        years=config['test_years'],
        window_size=config['window_size'],
        prediction_horizon=config['prediction_horizon'],
        sector=sector,
        resolution=config['resolution'],
        use_volume=True,
    )
    
    if len(train_dataset) < 100 or len(test_dataset) < 50:
        print(f"Insufficient data: train={len(train_dataset)}, test={len(test_dataset)}")
        return None
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    dist = train_dataset.get_class_distribution()
    print(f"Class distribution: {dist['pos_ratio']:.3f} positive")
    
    # Split train/val
    val_ratio = 0.15
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42 + run_id)
    )
    
    # DataLoaders with optimized settings for H20
    num_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_subset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=config['batch_size'],
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Create model
    print(f"Creating model: {config['model_type']}")
    if config['model_type'] == 'finance_h20':
        model = FinanceCNN_H20(
            window_size=config['window_size'],
            num_classes=2,
            resolution=config['resolution'],
            base_channels=config['base_channels'],
            num_blocks=config['num_blocks'],
            dropout=config['dropout'],
            use_attention=True,
        ).to(DEVICE)
    else:  # finance_large
        img_size = HighResStockDataset.RESOLUTIONS[config['resolution']][config['window_size']]
        model = FinanceCNN_Large(
            window_size=config['window_size'],
            num_classes=2,
            img_size=img_size,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=config['dropout'],
        ).to(DEVICE)
    
    param_count = model.count_parameters()
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss with class weighting
    class_weights = torch.tensor([1.0, dist['pos_ratio'] / (1 - dist['pos_ratio'])]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - use AdamW with cosine annealing
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    use_amp = config.get('use_amp', True)
    
    # Training loop
    best_f1 = 0
    best_state = None
    patience_counter = 0
    history = []
    
    print(f"\nTraining for max {config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch_amp(
            model, train_loader, criterion, optimizer, scaler, use_amp
        )
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
        })
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_metrics['f1']:.4f}/{val_metrics['accuracy']:.4f}/{val_metrics['auc']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Load best and evaluate
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    test_metrics = evaluate(model, test_loader, criterion)
    print(f"Test Results: F1={test_metrics['f1']:.4f}, Acc={test_metrics['accuracy']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    return {
        'sector': sector,
        'run_id': run_id,
        'test_metrics': test_metrics,
        'best_val_f1': best_f1,
        'epochs_trained': len(history),
        'history': history,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': len(test_dataset),
        'param_count': param_count,
        'training_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='H20 Finance CNN Training')
    parser.add_argument('--preset', type=str, default='high', 
                       choices=['standard', 'high', 'ultra', 'h20_max'],
                       help='Resolution preset')
    parser.add_argument('--window-size', type=int, default=20, choices=[20, 60])
    parser.add_argument('--prediction-horizon', type=int, default=5, choices=[5, 20, 60])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--sector', type=str, default=None)
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    args = parser.parse_args()
    
    # Get preset config
    config = deepcopy(H20_CONFIGS[args.preset])
    config['window_size'] = args.window_size
    config['prediction_horizon'] = args.prediction_horizon
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    config['dropout'] = args.dropout
    config['weight_decay'] = args.weight_decay
    config['patience'] = args.patience
    config['use_amp'] = not args.no_amp
    config['train_years'] = list(range(2018, 2022))
    config['test_years'] = [2022, 2023]
    
    print("="*80)
    print(f"H20 Finance CNN Training | Preset: {args.preset}")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Resolution: {config['resolution']}")
    print(f"Image size: {HighResStockDataset.RESOLUTIONS[config['resolution']][config['window_size']]}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Base channels: {config['base_channels']}")
    print(f"Num blocks: {config['num_blocks']}")
    print(f"Mixed precision: {config['use_amp']}")
    print(f"Window: {config['window_size']}d, Horizon: {config['prediction_horizon']}d")
    print()
    
    # Train
    sectors = [args.sector] if args.sector else list(US_STOCK_GROUPS.keys())
    all_results = {}
    
    for sector in sectors:
        sector_results = []
        for run_id in range(args.ensemble):
            result = train_sector_model(sector, config, run_id)
            if result:
                sector_results.append(result)
        
        if sector_results:
            avg_f1 = np.mean([r['test_metrics']['f1'] for r in sector_results])
            avg_acc = np.mean([r['test_metrics']['accuracy'] for r in sector_results])
            avg_auc = np.mean([r['test_metrics']['auc'] for r in sector_results])
            
            all_results[sector] = {
                'sector_name': US_STOCK_GROUPS[sector]['name'],
                'avg_f1': avg_f1,
                'avg_accuracy': avg_acc,
                'avg_auc': avg_auc,
                'ensemble_results': sector_results,
            }
            print(f"\n{sector}: F1={avg_f1:.4f}, Acc={avg_acc:.4f}, AUC={avg_auc:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = OUTPUT_DIR / f"h20_{args.preset}_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'config': config,
            'preset': args.preset,
            'results': all_results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved: {result_file}")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if all_results:
        f1_scores = [r['avg_f1'] for r in all_results.values()]
        print(f"\nAverage F1: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
        print(f"Best sector: {max(all_results.items(), key=lambda x: x[1]['avg_f1'])[0]}")
        print(f"Worst sector: {min(all_results.items(), key=lambda x: x[1]['avg_f1'])[0]}")


if __name__ == '__main__':
    main()
