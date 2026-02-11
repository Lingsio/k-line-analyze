"""
Vision Transformer (ViT) 实验脚本
使用ViT对K线图像进行分类预测
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
from src.models.vision_transformer import VisionTransformerSmall, VisionTransformerTiny

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = min(8, multiprocessing.cpu_count())
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()

CONFIG = {
    'model_type': 'vit-small',  # 'vit-tiny', 'vit-small'
    'window_size': 20,
    'prediction_horizon': 5,
    'img_size': (128, 128),
    'patch_size': 16,
    'epochs': 20,
    'batch_size': 128,
    'lr': 3e-4,  # ViT通常用较小学习率
    'weight_decay': 1e-4,
    'patience': 5,
    'num_runs': 3,
    'norm_method': 'minmax',
    'use_clahe': False,
    'output_channels': 'rgb',
    'augment_prob': 0.3,
    'label_threshold': 0.005,
}

# Data directories
DATA_DIRS = [
    str(PROJECT_ROOT / 'data' / 'raw')
]

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eccv_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def evaluate_model(model, dataloader, device, use_amp=False):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, _, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=use_amp):
                outputs = model(imgs)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }

    return metrics


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for imgs, _, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if torch.isnan(loss):
            print("  Warning: NaN loss detected, skipping batch")
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
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)

    return avg_loss, train_acc


def run_single_experiment(seed, model_type='vit-small'):
    """运行单次实验"""
    print(f"\n{'='*70}")
    print(f"Starting run with seed={seed}, model={model_type}")
    print(f"{'='*70}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = StockDataset(
        data_dir=DATA_DIRS,
        mode='train',
        window_size=CONFIG['window_size'],
        prediction_horizon=CONFIG['prediction_horizon'],
        img_size=CONFIG['img_size'],
        norm_method=CONFIG['norm_method'],
        use_clahe=CONFIG['use_clahe'],
        output_channels=CONFIG['output_channels'],
        augment_prob=CONFIG['augment_prob'],
        label_threshold=CONFIG['label_threshold'],
    )

    val_dataset = StockDataset(
        data_dir=DATA_DIRS,
        mode='val',
        window_size=CONFIG['window_size'],
        prediction_horizon=CONFIG['prediction_horizon'],
        img_size=CONFIG['img_size'],
        norm_method=CONFIG['norm_method'],
        use_clahe=CONFIG['use_clahe'],
        output_channels=CONFIG['output_channels'],
        augment_prob=0.0,
        label_threshold=CONFIG['label_threshold'],
    )

    test_dataset = StockDataset(
        data_dir=DATA_DIRS,
        mode='test',
        window_size=CONFIG['window_size'],
        prediction_horizon=CONFIG['prediction_horizon'],
        img_size=CONFIG['img_size'],
        norm_method=CONFIG['norm_method'],
        use_clahe=CONFIG['use_clahe'],
        output_channels=CONFIG['output_channels'],
        augment_prob=0.0,
        label_threshold=CONFIG['label_threshold'],
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )

    # Create model
    print(f"\nCreating {model_type} model...")
    if model_type == 'vit-tiny':
        model = VisionTransformerTiny(
            img_size=CONFIG['img_size'][0],
            patch_size=CONFIG['patch_size'],
            in_channels=3,
            num_classes=2
        )
    else:  # vit-small
        model = VisionTransformerSmall(
            img_size=CONFIG['img_size'][0],
            patch_size=CONFIG['patch_size'],
            in_channels=3,
            num_classes=2
        )

    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")

    # Training setup
    class_weights = train_dataset.get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = GradScaler() if USE_AMP else None

    # Training loop
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    history = []

    print("\nStarting training...")
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, USE_AMP
        )

        val_metrics = evaluate_model(model, val_loader, DEVICE, USE_AMP)
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1']

        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_f1': val_f1
        })

        print(f"Epoch {epoch:2d}/{CONFIG['epochs']}: "
              f"Loss={train_loss:.4f}, "
              f"TrainAcc={train_acc:.4f}, "
              f"ValAcc={val_acc:.4f}, "
              f"ValF1={val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model and evaluate
    print("\nEvaluating best model on test set...")
    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, DEVICE, USE_AMP)

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")

    return {
        'seed': seed,
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'epochs_trained': epoch,
        'history': history
    }


def main():
    print("="*70)
    print("Vision Transformer (ViT) Experiment")
    print("="*70)
    print(f"Config: {CONFIG}")
    print(f"Device: {DEVICE}")
    print(f"Use AMP: {USE_AMP}")
    print("="*70)

    results = []
    seeds = [42, 142, 242][:CONFIG['num_runs']]

    for seed in seeds:
        result = run_single_experiment(seed, CONFIG['model_type'])
        results.append(result)

    # Aggregate results
    test_accs = [r['test_metrics']['accuracy'] for r in results]
    test_f1s = [r['test_metrics']['f1'] for r in results]
    test_aucs = [r['test_metrics']['auc'] for r in results]

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"F1 Score: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"AUC:      {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print("="*70)

    # Save results
    output = {
        'config': CONFIG,
        'device': str(DEVICE),
        'model_type': CONFIG['model_type'],
        'num_runs': len(results),
        'aggregate_metrics': {
            'accuracy': {'mean': float(np.mean(test_accs)), 'std': float(np.std(test_accs))},
            'f1': {'mean': float(np.mean(test_f1s)), 'std': float(np.std(test_f1s))},
            'auc': {'mean': float(np.mean(test_aucs)), 'std': float(np.std(test_aucs))},
        },
        'individual_runs': results,
        'timestamp': datetime.now().isoformat()
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'vit_{CONFIG["model_type"]}_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
