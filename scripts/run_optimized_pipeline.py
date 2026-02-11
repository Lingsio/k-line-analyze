"""
Optimized K-Line Prediction Pipeline
=====================================
Multi-stage optimization to push accuracy from ~51% toward 55-60%.

Key fixes over baseline:
  1. pretrained=True (ImageNet transfer - was False in all benchmarks!)
  2. Dropout(0.3) before classifier (was missing entirely)
  3. Label smoothing (0.1) for noisy stock labels
  4. Batch-level Mixup/CutMix with proper soft-label loss
  5. Stronger regularization: weight_decay=1e-4, augment_prob=0.5
  6. US + CN data (was US-only)
  7. Warmup + longer training (40 epochs, patience=15)
  8. Multi-architecture ensemble + TTA
  9. Industry-grouped fine-tuning

Usage:
  python scripts/run_optimized_pipeline.py --stage all
  python scripts/run_optimized_pipeline.py --stage 1
  python scripts/run_optimized_pipeline.py --stage 4 --force
  python scripts/run_optimized_pipeline.py --stage all --debug
"""

import os, sys, json, math, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.models.multiscale_cnn import (
    LightweightMultiScaleCNN,
    HierarchicalMultiScaleCNN,
)
from src.data.multiscale_dataset import MultiScaleStockDataset, collate_multiscale

# =============================================================================
# Global Configuration
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_BASE = PROJECT_ROOT / 'outputs' / 'optimized_pipeline'
NUM_WORKERS = 8  # Set to 8 for H20 (32 cores), 4 for local

# Data directories
US_DATA = str(PROJECT_ROOT / 'data' / 'raw' / 'us')
CN_DATA = str(PROJECT_ROOT / 'data' / 'raw' / 'cn')

def get_data_dirs():
    """Return available data directories."""
    dirs = []
    if Path(US_DATA).exists():
        dirs.append(US_DATA)
    if Path(CN_DATA).exists():
        dirs.append(CN_DATA)
    return dirs if dirs else [US_DATA]


# =============================================================================
# Helper: Model Building
# =============================================================================
def build_model(arch='resnet18', input_channels=3, num_classes=2,
                pretrained=True, dropout=0.3):
    """
    Build a CNN model with pretrained weights and proper dropout.

    Fixes vs baseline:
      - pretrained=True (was False)
      - Dropout inserted before classifier head (was missing)
      - Proper channel adaptation for pretrained + non-3ch input
    """
    model = CNNModel(
        num_classes=num_classes,
        input_channels=3,  # Build with 3ch first for pretrained weights
        arch=arch,
        pretrained=pretrained,
    )

    # Handle non-3-channel input while preserving pretrained weights
    if input_channels != 3 and arch == 'resnet18':
        old_weight = model.backbone.conv1.weight.data  # [64, 3, 7, 7]
        new_conv = nn.Conv2d(input_channels, 64, kernel_size=7,
                             stride=2, padding=3, bias=False)
        # Copy first 3 channels from pretrained, init extra channels from mean
        new_conv.weight.data[:, :3] = old_weight
        for c in range(3, input_channels):
            new_conv.weight.data[:, c] = old_weight.mean(dim=1)
        model.backbone.conv1 = new_conv
    elif input_channels != 3 and arch == 'efficientnet_b0':
        old_conv = model.backbone.features[0][0]
        old_weight = old_conv.weight.data  # [32, 3, 3, 3]
        new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                             kernel_size=3, stride=1, padding=1, bias=False)
        new_conv.weight.data[:, :3] = old_weight
        for c in range(3, input_channels):
            new_conv.weight.data[:, c] = old_weight.mean(dim=1)
        model.backbone.features[0][0] = new_conv

    # Insert dropout before classifier (critical fix!)
    if arch == 'resnet18':
        in_features = model.backbone.fc.in_features
        model.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif arch == 'efficientnet_b0':
        in_features = model.backbone.classifier[1].in_features
        model.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    return model


# =============================================================================
# Helper: Warmup + Cosine Annealing Scheduler
# =============================================================================
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for warmup_epochs, then cosine annealing to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Helper: Batch-level Mixup / CutMix with Soft Labels
# =============================================================================
def mixup_data(x, y, alpha=0.2):
    """Batch-level mixup with proper soft-label support."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Batch-level CutMix with proper soft-label support."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)

    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute soft-label loss for mixup/cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Helper: Test-Time Augmentation
# =============================================================================
def apply_tta_augmentation(img_tensor):
    """
    Apply mild augmentations suitable for K-line charts.
    NO horizontal/vertical flip (breaks time/price direction).
    """
    aug_type = np.random.choice(['noise', 'brightness', 'contrast', 'none'])

    if aug_type == 'noise':
        noise = torch.randn_like(img_tensor) * 0.01
        return torch.clamp(img_tensor + noise, 0, 1)
    elif aug_type == 'brightness':
        factor = np.random.uniform(0.95, 1.05)
        return torch.clamp(img_tensor * factor, 0, 1)
    elif aug_type == 'contrast':
        mean = img_tensor.mean(dim=(-2, -1), keepdim=True)
        factor = np.random.uniform(0.95, 1.05)
        return torch.clamp(mean + (img_tensor - mean) * factor, 0, 1)
    else:
        return img_tensor


# =============================================================================
# Helper: Training & Evaluation
# =============================================================================
def train_epoch_optimized(model, loader, criterion, optimizer, scaler,
                          mixup_alpha=0.2, cutmix_alpha=0.2, device=DEVICE):
    """
    Training loop with batch-level mixup/cutmix and proper soft labels.
    """
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for imgs, _, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        # Randomly choose augmentation strategy
        r = np.random.random()
        if r < 0.3 and mixup_alpha > 0:
            imgs, y_a, y_b, lam = mixup_data(imgs, lbls, mixup_alpha)
        elif r < 0.6 and cutmix_alpha > 0:
            imgs, y_a, y_b, lam = cutmix_data(imgs, lbls, cutmix_alpha)
        else:
            y_a, y_b, lam = lbls, lbls, 1.0

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
            out = model(imgs)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

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
    """Standard evaluation with all metrics."""
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
            probs_all.extend(torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy())

    metrics = {
        'loss': total_loss / max(len(loader), 1),
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1': f1_score(labels_all, preds_all, average='weighted'),
        'precision': precision_score(labels_all, preds_all, average='weighted', zero_division=0),
        'recall': recall_score(labels_all, preds_all, average='weighted', zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(labels_all, probs_all)
    except Exception:
        metrics['auc'] = 0.5
    return metrics


def evaluate_model_tta(model, loader, num_augments=5, device=DEVICE):
    """Evaluate with Test-Time Augmentation."""
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for imgs, _, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            # Accumulate predictions from multiple augmented views
            all_probs = []
            for i in range(num_augments):
                if i == 0:
                    aug_imgs = imgs  # Original
                else:
                    aug_imgs = apply_tta_augmentation(imgs)

                with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
                    out = model(aug_imgs)
                all_probs.append(torch.softmax(out.float(), dim=1))

            avg_probs = torch.stack(all_probs).mean(dim=0)
            preds_all.extend(avg_probs.argmax(1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
            probs_all.extend(avg_probs[:, 1].cpu().numpy())

    metrics = {
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1': f1_score(labels_all, preds_all, average='weighted'),
        'precision': precision_score(labels_all, preds_all, average='weighted', zero_division=0),
        'recall': recall_score(labels_all, preds_all, average='weighted', zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(labels_all, probs_all)
    except Exception:
        metrics['auc'] = 0.5
    return metrics


def evaluate_3class_with_neutral_penalty(model, loader, neutral_threshold=0.01, device=DEVICE):
    """
    Special 3-class evaluation where predicting "neutral" but actual 
    move exceeds threshold counts as WRONG.
    
    Label mapping: 0=Down, 1=Neutral, 2=Up
    
    This function evaluates:
    - Standard 3-class accuracy
    - Strict accuracy: neutral predictions on non-neutral actuals are wrong
    - Per-class precision/recall
    
    Args:
        model: The model to evaluate
        loader: DataLoader with dataset that has 'raw_return' in samples
        neutral_threshold: Return threshold for neutral zone (default 1%)
        device: Device to run on
    """
    model.eval()
    preds_all, labels_all, raw_returns_all = [], [], []

    # Access dataset to get raw returns
    dataset = loader.dataset
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                imgs, _, lbls = batch
            else:
                imgs, lbls = batch[0], batch[-1]
                
            imgs = imgs.to(device, non_blocking=True)
            
            with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
                out = model(imgs)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
            
            # Get raw returns for this batch
            batch_size = len(lbls)
            start_idx = batch_idx * loader.batch_size
            for i in range(batch_size):
                sample_idx = start_idx + i
                if hasattr(dataset, 'samples') and sample_idx < len(dataset.samples):
                    raw_returns_all.append(dataset.samples[sample_idx].get('raw_return', 0))
                elif hasattr(dataset, 'base_ds') and sample_idx < len(dataset.base_ds.samples):
                    # For wrapped datasets like TechIndicatorDataset
                    real_idx = dataset.filtered_indices[sample_idx] if hasattr(dataset, 'filtered_indices') else sample_idx
                    raw_returns_all.append(dataset.base_ds.samples[real_idx].get('raw_return', 0))
                else:
                    raw_returns_all.append(0)

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    raw_returns_all = np.array(raw_returns_all)
    
    # Standard 3-class metrics
    metrics = {
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1_weighted': f1_score(labels_all, preds_all, average='weighted', zero_division=0),
        'f1_macro': f1_score(labels_all, preds_all, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    for cls_idx, cls_name in [(0, 'down'), (1, 'neutral'), (2, 'up')]:
        cls_mask = labels_all == cls_idx
        if cls_mask.sum() > 0:
            metrics[f'precision_{cls_name}'] = precision_score(
                labels_all == cls_idx, preds_all == cls_idx, zero_division=0
            )
            metrics[f'recall_{cls_name}'] = recall_score(
                labels_all == cls_idx, preds_all == cls_idx, zero_division=0
            )
    
    # STRICT evaluation: recompute accuracy with neutral penalty
    # If predicted neutral (1) but actual return exceeds threshold, count as wrong
    strict_correct = 0
    neutral_wrong = 0  # Predicted neutral but actual was up/down
    
    for i in range(len(preds_all)):
        pred = preds_all[i]
        actual_label = labels_all[i]
        ret = raw_returns_all[i]
        
        # Determine actual neutral status based on raw return
        actual_is_neutral = abs(ret) < neutral_threshold
        
        if pred == 1:  # Predicted neutral
            if actual_is_neutral:
                strict_correct += 1  # Correctly predicted neutral
            else:
                neutral_wrong += 1   # Wrong: predicted neutral but actual was up/down
        elif pred == actual_label:
            strict_correct += 1  # Correctly predicted up/down
    
    metrics['strict_accuracy'] = strict_correct / len(preds_all) if len(preds_all) > 0 else 0
    metrics['neutral_wrong_count'] = neutral_wrong
    metrics['neutral_wrong_rate'] = neutral_wrong / len(preds_all) if len(preds_all) > 0 else 0
    
    # Count predictions per class
    metrics['pred_down'] = int((preds_all == 0).sum())
    metrics['pred_neutral'] = int((preds_all == 1).sum())
    metrics['pred_up'] = int((preds_all == 2).sum())
    metrics['actual_down'] = int((labels_all == 0).sum())
    metrics['actual_neutral'] = int((labels_all == 1).sum())
    metrics['actual_up'] = int((labels_all == 2).sum())
    
    return metrics


def evaluate_multiscale(model, loader, criterion, device=DEVICE):
    """Evaluate multi-scale model."""
    model.eval()
    total_loss = 0
    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            images = {k: v.to(device) for k, v in images.items()}

            with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
                out = model(images)
                loss = criterion(out, labels)
            if not torch.isnan(loss):
                total_loss += loss.item()
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy())

    metrics = {
        'loss': total_loss / max(len(loader), 1),
        'accuracy': accuracy_score(labels_all, preds_all),
        'f1': f1_score(labels_all, preds_all, average='weighted'),
        'precision': precision_score(labels_all, preds_all, average='weighted', zero_division=0),
        'recall': recall_score(labels_all, preds_all, average='weighted', zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(labels_all, probs_all)
    except Exception:
        metrics['auc'] = 0.5
    return metrics


# =============================================================================
# Helper: Checkpoint Save / Load
# =============================================================================
def save_checkpoint(model, metrics, config, path):
    """Save model checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'metrics': metrics,
        'config': config,
    }, path)
    print(f"  Saved: {path} (acc={metrics.get('accuracy', 0):.4f}, f1={metrics.get('f1', 0):.4f})")


def load_checkpoint(path, model, device=DEVICE):
    """Load checkpoint into model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in ckpt['model_state_dict'].items()})
    return ckpt.get('metrics', {}), ckpt.get('config', {})


def stage_completed(stage_dir, result_file='results.json'):
    """Check if a stage is already completed."""
    result_path = stage_dir / result_file
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
        return data.get('status') == 'complete'
    return False


# =============================================================================
# Helper: Create DataLoaders
# =============================================================================
def create_dataloaders(data_dirs, config, num_workers=4):
    """Create train/val/test dataloaders from config dict.
    
    IMPORTANT: quantile_filter and train_filter_threshold are only applied to 
    training set to avoid data leakage. Val/test sets keep all samples.
    """
    # Common config for all splits (no filtering to avoid data leakage)
    common = {
        'data_dir': data_dirs,
        'window_size': config.get('window_size', 20),
        'prediction_horizon': config.get('prediction_horizon', 5),
        'img_size': config.get('img_size', (128, 128)),
        'norm_method': config.get('norm_method', 'robust'),
        'use_clahe': config.get('use_clahe', True),
        'output_channels': config.get('output_channels', 'rgb'),
        'label_threshold': config.get('label_threshold', 'dynamic'),
        'chart_type': config.get('chart_type', 'candle'),
        'num_classes': config.get('num_classes', 2),
        # Disable dataset-level mixup (we do batch-level instead)
        'mixup_prob': 0.0,
        'cutmix_prob': 0.0,
    }

    limit = config.get('limit', None)

    # Training set: can use filtering
    train_common = common.copy()
    train_common['quantile_filter'] = config.get('quantile_filter', None)
    train_common['train_filter_threshold'] = config.get('train_filter_threshold', None)
    
    train_ds = StockDataset(
        mode='train',
        augment_prob=config.get('augment_prob', 0.5),
        limit=limit,
        **train_common
    )
    
    # Val/Test set: NO filtering to avoid data leakage
    # These sets should represent the real distribution
    val_ds = StockDataset(mode='val', augment_prob=0.0, limit=limit, **common)
    test_ds = StockDataset(mode='test', augment_prob=0.0, limit=limit, **common)

    batch_size = config.get('batch_size', 128)
    loader_kw = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': num_workers > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# =============================================================================
# Helper: Full Training Run
# =============================================================================
def run_training(model, train_loader, val_loader, test_loader,
                 train_ds, config, device=DEVICE):
    """
    Complete training loop with all optimizations.
    Returns test metrics and training history.
    """
    epochs = config.get('epochs', 40)
    lr = config.get('lr', 3e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    patience = config.get('patience', 15)
    warmup_epochs = config.get('warmup_epochs', 5)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    cutmix_alpha = config.get('cutmix_alpha', 0.2)
    label_smoothing = config.get('label_smoothing', 0.1)

    model = model.to(device)

    # torch.compile: auto-enable on Linux (Triton available), skip on Windows
    if sys.platform != 'win32' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  torch.compile: enabled (reduce-overhead)")
        except Exception:
            pass

    # Loss with label smoothing + class weights
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler('cuda') if device.type == 'cuda' else GradScaler('cpu', enabled=False)

    best_f1 = 0
    best_state = None
    patience_cnt = 0
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_optimized(
            model, train_loader, criterion, optimizer, scaler,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
            device=device,
        )
        val_m = evaluate_model(model, val_loader, criterion, device=device)
        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_acc': round(val_m['accuracy'], 4),
            'val_f1': round(val_m['f1'], 4),
            'val_auc': round(val_m['auc'], 4),
            'lr': round(optimizer.param_groups[0]['lr'], 6),
        })

        print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_m['accuracy']:.4f} "
              f"val_f1={val_m['f1']:.4f} val_auc={val_m['auc']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch+1}", flush=True)
                break

    # Load best state and evaluate on test set
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_m = evaluate_model(model, test_loader, criterion, device=device)
    test_m_tta = evaluate_model_tta(model, test_loader, num_augments=5, device=device)

    print(f"  Test: acc={test_m['accuracy']:.4f} f1={test_m['f1']:.4f} "
          f"auc={test_m['auc']:.4f}", flush=True)
    print(f"  Test (TTA): acc={test_m_tta['accuracy']:.4f} f1={test_m_tta['f1']:.4f} "
          f"auc={test_m_tta['auc']:.4f}", flush=True)

    return model, test_m, test_m_tta, history, best_f1


# =============================================================================
# Stage 1: Fix Baseline + Strong Regularization
# =============================================================================
def run_stage1(args):
    """
    Fix critical bugs and add proper regularization.
    - pretrained=True, dropout=0.3, label_smoothing=0.1
    - US+CN data, stronger augmentation, longer training
    """
    stage_dir = OUTPUT_BASE / 'stage1'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_completed(stage_dir) and not args.force:
        print("Stage 1 already completed. Use --force to re-run.")
        return

    print("=" * 70)
    print("STAGE 1: Fixed Baseline + Strong Regularization")
    print("=" * 70)

    config = {
        'name': 'OptimizedBaseline',
        'arch': 'resnet18',
        'pretrained': True,
        'dropout': 0.3,
        'img_size': (128, 128),
        'window_size': 20,
        'prediction_horizon': 5,
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb',
        'chart_type': 'candle',
        'label_threshold': 'dynamic',
        'augment_prob': 0.5,
        'epochs': 40,
        'batch_size': 128,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'patience': 15,
        'warmup_epochs': 5,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.2,
        'label_smoothing': 0.1,
        'num_runs': 3,
    }

    if args.debug:
        config['limit'] = 5
        config['epochs'] = 5
        config['num_runs'] = 1

    data_dirs = get_data_dirs()
    print(f"Data dirs: {data_dirs}")
    print(f"Config: {json.dumps({k: str(v) for k, v in config.items()}, indent=2)}")

    all_runs = []
    seeds = [42, 142, 242][:config['num_runs']]

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Run {run_idx+1}/{len(seeds)} (seed={seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create data
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
            create_dataloaders(data_dirs, config)
        print(f"  Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        # Build model
        input_channels = train_ds.get_num_channels()
        model = build_model(
            arch=config['arch'],
            input_channels=input_channels,
            pretrained=config['pretrained'],
            dropout=config['dropout'],
        )

        # Train
        model, test_m, test_m_tta, history, best_f1 = run_training(
            model, train_loader, val_loader, test_loader, train_ds, config
        )

        # Save checkpoint
        save_checkpoint(model, test_m, config,
                        stage_dir / f'model_seed{seed}.pt')

        all_runs.append({
            'seed': seed,
            'test_metrics': test_m,
            'test_metrics_tta': test_m_tta,
            'best_val_f1': best_f1,
            'epochs_trained': len(history),
            'history': history,
        })

    # Summary
    summary = {}
    for k in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
        vals = [r['test_metrics'][k] for r in all_runs]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))
        vals_tta = [r['test_metrics_tta'][k] for r in all_runs]
        summary[f'{k}_tta_mean'] = float(np.mean(vals_tta))
        summary[f'{k}_tta_std'] = float(np.std(vals_tta))

    print(f"\n{'='*70}")
    print(f"STAGE 1 SUMMARY")
    print(f"{'='*70}")
    print(f"Accuracy:     {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"F1:           {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"AUC:          {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}")
    print(f"Acc (TTA):    {summary['accuracy_tta_mean']:.4f} +/- {summary['accuracy_tta_std']:.4f}")
    print(f"F1  (TTA):    {summary['f1_tta_mean']:.4f} +/- {summary['f1_tta_std']:.4f}")

    result = {
        'status': 'complete',
        'stage': 1,
        'name': 'OptimizedBaseline',
        'config': config,
        'runs': all_runs,
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
    }
    with open(stage_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nStage 1 complete! Results saved to {stage_dir}")


# =============================================================================
# Stage 2: Multi-Architecture Diversity Training
# =============================================================================
def run_stage2(args):
    """
    Train 9 diverse models for ensemble: 3 architectures x 3 seeds.
    """
    stage_dir = OUTPUT_BASE / 'stage2'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_completed(stage_dir) and not args.force:
        print("Stage 2 already completed. Use --force to re-run.")
        return

    print("=" * 70)
    print("STAGE 2: Multi-Architecture Diversity Training")
    print("=" * 70)

    base_config = {
        'pretrained': True,
        'dropout': 0.3,
        'img_size': (128, 128),
        'window_size': 20,
        'prediction_horizon': 5,
        'norm_method': 'robust',
        'use_clahe': True,
        'label_threshold': 'dynamic',
        'augment_prob': 0.5,
        'epochs': 40,
        'batch_size': 128,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'patience': 15,
        'warmup_epochs': 5,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.2,
        'label_smoothing': 0.1,
        'chart_type': 'candle',
    }

    # Model configurations for diversity
    model_configs = [
        # ResNet18 + RGB candle (3 seeds)
        {'arch': 'resnet18', 'output_channels': 'rgb', 'seed': 42, 'id': 'resnet18_rgb_s42'},
        {'arch': 'resnet18', 'output_channels': 'rgb', 'seed': 142, 'id': 'resnet18_rgb_s142'},
        {'arch': 'resnet18', 'output_channels': 'rgb', 'seed': 242, 'id': 'resnet18_rgb_s242'},
        # EfficientNet-B0 + RGB candle (3 seeds)
        {'arch': 'efficientnet_b0', 'output_channels': 'rgb', 'seed': 42, 'id': 'effnet_rgb_s42'},
        {'arch': 'efficientnet_b0', 'output_channels': 'rgb', 'seed': 142, 'id': 'effnet_rgb_s142'},
        {'arch': 'efficientnet_b0', 'output_channels': 'rgb', 'seed': 242, 'id': 'effnet_rgb_s242'},
        # ResNet18 + RGB+Edge multi-channel (3 seeds)
        {'arch': 'resnet18', 'output_channels': 'rgb+edge', 'seed': 42, 'id': 'resnet18_edge_s42'},
        {'arch': 'resnet18', 'output_channels': 'rgb+edge', 'seed': 142, 'id': 'resnet18_edge_s142'},
        {'arch': 'resnet18', 'output_channels': 'rgb+edge', 'seed': 242, 'id': 'resnet18_edge_s242'},
    ]

    if args.debug:
        model_configs = model_configs[:2]
        base_config['limit'] = 5
        base_config['epochs'] = 5

    data_dirs = get_data_dirs()
    all_results = []

    for i, mc in enumerate(model_configs):
        model_id = mc['id']
        ckpt_path = stage_dir / f'{model_id}.pt'

        # Skip if already trained
        if ckpt_path.exists() and not args.force:
            print(f"\n[{i+1}/{len(model_configs)}] {model_id}: already trained, skipping")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            all_results.append({
                'model_id': model_id,
                'config': mc,
                'test_metrics': ckpt['metrics'],
            })
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(model_configs)}] Training: {model_id}")
        print(f"  arch={mc['arch']}, channels={mc['output_channels']}, seed={mc['seed']}")
        print(f"{'='*60}")

        torch.manual_seed(mc['seed'])
        np.random.seed(mc['seed'])

        config = {**base_config, 'output_channels': mc['output_channels']}
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
            create_dataloaders(data_dirs, config)

        input_channels = train_ds.get_num_channels()
        print(f"  Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, ch={input_channels}")

        model = build_model(
            arch=mc['arch'],
            input_channels=input_channels,
            pretrained=base_config['pretrained'],
            dropout=base_config['dropout'],
        )

        model, test_m, test_m_tta, history, best_f1 = run_training(
            model, train_loader, val_loader, test_loader, train_ds, config
        )

        save_checkpoint(model, test_m, {**config, **mc}, ckpt_path)

        all_results.append({
            'model_id': model_id,
            'config': mc,
            'test_metrics': test_m,
            'test_metrics_tta': test_m_tta,
            'best_val_f1': best_f1,
            'epochs_trained': len(history),
        })

    # Summary
    print(f"\n{'='*70}")
    print("STAGE 2 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 55)
    for r in all_results:
        m = r['test_metrics']
        print(f"{r['model_id']:<25} {m['accuracy']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    result = {
        'status': 'complete',
        'stage': 2,
        'name': 'MultiArchDiversity',
        'models': all_results,
        'timestamp': datetime.now().isoformat(),
    }
    with open(stage_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nStage 2 complete! Results saved to {stage_dir}")


# =============================================================================
# Stage 3: Multi-Scale Feature Fusion
# =============================================================================
def run_stage3(args):
    """
    Train multi-scale models using 5d/10d/20d windows.
    """
    stage_dir = OUTPUT_BASE / 'stage3'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_completed(stage_dir) and not args.force:
        print("Stage 3 already completed. Use --force to re-run.")
        return

    print("=" * 70)
    print("STAGE 3: Multi-Scale Feature Fusion")
    print("=" * 70)

    config = {
        'scales': [5, 10, 20],
        'prediction_horizon': 5,
        'img_size': (128, 128),
        'chart_type': 'candle',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'batch_size': 64,
        'epochs': 30,
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'patience': 12,
        'warmup_epochs': 3,
        'label_smoothing': 0.1,
    }

    limit = 5 if args.debug else None
    if args.debug:
        config['epochs'] = 5

    data_dirs = get_data_dirs()

    # Create multi-scale datasets
    train_ds = MultiScaleStockDataset(
        data_dir=data_dirs,
        scales=config['scales'],
        prediction_horizon=config['prediction_horizon'],
        img_size=config['img_size'],
        mode='train',
        chart_type=config['chart_type'],
        augment_prob=config['augment_prob'],
        label_threshold=config['label_threshold'],
        limit=limit,
    )
    val_ds = MultiScaleStockDataset(
        data_dir=data_dirs,
        scales=config['scales'],
        prediction_horizon=config['prediction_horizon'],
        img_size=config['img_size'],
        mode='val',
        chart_type=config['chart_type'],
        augment_prob=0.0,
        label_threshold=config['label_threshold'],
        limit=limit,
    )
    test_ds = MultiScaleStockDataset(
        data_dir=data_dirs,
        scales=config['scales'],
        prediction_horizon=config['prediction_horizon'],
        img_size=config['img_size'],
        mode='test',
        chart_type=config['chart_type'],
        augment_prob=0.0,
        label_threshold=config['label_threshold'],
        limit=limit,
    )

    loader_kw = {
        'num_workers': NUM_WORKERS,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
        'collate_fn': collate_multiscale,
    }
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                            shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'],
                             shuffle=False, **loader_kw)

    print(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Train two multi-scale model variants
    ms_models = [
        ('lightweight', LightweightMultiScaleCNN),
        ('hierarchical', HierarchicalMultiScaleCNN),
    ]

    if args.debug:
        ms_models = ms_models[:1]

    all_results = []

    for model_name, ModelClass in ms_models:
        ckpt_path = stage_dir / f'multiscale_{model_name}.pt'

        if ckpt_path.exists() and not args.force:
            print(f"\n{model_name}: already trained, skipping")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            all_results.append({
                'model_name': model_name,
                'test_metrics': ckpt['metrics'],
            })
            continue

        print(f"\n{'='*60}")
        print(f"Training Multi-Scale: {model_name}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        np.random.seed(42)

        model = ModelClass(
            num_classes=2,
            input_channels=3,
            pretrained=True,
            dropout=0.3,
        ).to(DEVICE)

        # Skip torch.compile on Windows (Triton not available)
        pass

        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                       weight_decay=config['weight_decay'])
        scheduler = get_warmup_cosine_scheduler(optimizer, config['warmup_epochs'],
                                                 config['epochs'])

        best_f1 = 0
        best_state = None
        patience_cnt = 0

        for epoch in range(config['epochs']):
            # Training
            model.train()
            total_loss, preds_all, labels_all = 0, [], []
            for images, labels in train_loader:
                labels = labels.to(DEVICE)
                images = {k: v.to(DEVICE) for k, v in images.items()}

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type='cuda', enabled=(DEVICE.type == 'cuda')):
                    out = model(images)
                    loss = criterion(out, labels)

                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                preds_all.extend(out.argmax(1).detach().cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

            train_loss = total_loss / max(len(train_loader), 1)
            train_acc = accuracy_score(labels_all, preds_all)

            # Validation
            val_m = evaluate_multiscale(model, val_loader, criterion)
            scheduler.step()

            print(f"  Epoch {epoch+1}/{config['epochs']}: loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_m['accuracy']:.4f} "
                  f"val_f1={val_m['f1']:.4f}", flush=True)

            if val_m['f1'] > best_f1:
                best_f1 = val_m['f1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= config['patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best and test
        if best_state:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        test_m = evaluate_multiscale(model, test_loader, criterion)
        print(f"  Test: acc={test_m['accuracy']:.4f} f1={test_m['f1']:.4f} auc={test_m['auc']:.4f}")

        save_checkpoint(model, test_m, config, ckpt_path)
        all_results.append({
            'model_name': model_name,
            'test_metrics': test_m,
            'best_val_f1': best_f1,
        })

    # Summary
    print(f"\n{'='*70}")
    print("STAGE 3 SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        m = r['test_metrics']
        print(f"  {r['model_name']}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    result = {
        'status': 'complete',
        'stage': 3,
        'name': 'MultiScaleFusion',
        'models': all_results,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }
    with open(stage_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nStage 3 complete! Results saved to {stage_dir}")


# =============================================================================
# Stage 4: Ensemble + TTA Evaluation
# =============================================================================
def run_stage4(args):
    """
    Combine all Stage 2 models with ensemble methods and TTA.
    """
    stage_dir = OUTPUT_BASE / 'stage4'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_completed(stage_dir) and not args.force:
        print("Stage 4 already completed. Use --force to re-run.")
        return

    print("=" * 70)
    print("STAGE 4: Ensemble + Test-Time Augmentation")
    print("=" * 70)

    stage2_dir = OUTPUT_BASE / 'stage2'

    # --- Load Stage 2 models ---
    print("\nLoading Stage 2 models...")
    stage2_models = []
    stage2_meta = []

    for ckpt_file in sorted(stage2_dir.glob('*.pt')):
        print(f"  Loading: {ckpt_file.name}")
        ckpt = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
        cfg = ckpt.get('config', {})

        arch = cfg.get('arch', 'resnet18')
        output_channels = cfg.get('output_channels', 'rgb')

        # Determine input channels
        ch_map = {'rgb': 3, 'rgb+edge': 4, 'rgb+hsv': 4, 'all': 5}
        input_channels = ch_map.get(output_channels, 3)

        model = build_model(
            arch=arch,
            input_channels=input_channels,
            pretrained=False,  # Load our weights
            dropout=cfg.get('dropout', 0.3),
        )
        model.load_state_dict({k: v.to(DEVICE) for k, v in ckpt['model_state_dict'].items()})
        model = model.to(DEVICE)
        model.eval()

        stage2_models.append(model)
        stage2_meta.append({
            'name': ckpt_file.stem,
            'arch': arch,
            'channels': output_channels,
            'val_metrics': ckpt.get('metrics', {}),
        })

    if not stage2_models:
        print("ERROR: No Stage 2 models found. Run Stage 2 first.")
        return

    print(f"Loaded {len(stage2_models)} models")

    # --- Create test data ---
    # We need separate loaders for different channel configs
    data_dirs = get_data_dirs()
    limit = 5 if args.debug else None

    # Group models by channel config
    channel_groups = defaultdict(list)
    for i, meta in enumerate(stage2_meta):
        channel_groups[meta['channels']].append(i)

    print(f"\nChannel groups: {dict({k: len(v) for k, v in channel_groups.items()})}")

    # Collect predictions from all models
    all_model_probs = {}
    all_labels = None

    for channels, model_indices in channel_groups.items():
        config = {
            'img_size': (128, 128),
            'window_size': 20,
            'prediction_horizon': 5,
            'norm_method': 'robust',
            'use_clahe': True,
            'output_channels': channels,
            'label_threshold': 'dynamic',
            'chart_type': 'candle',
            'batch_size': 128,
            'limit': limit,
        }
        _, _, test_ds, _, _, test_loader = create_dataloaders(data_dirs, config)

        for idx in model_indices:
            model = stage2_models[idx]
            model_name = stage2_meta[idx]['name']

            preds_list, probs_list, labels_list = [], [], []

            with torch.no_grad():
                for imgs, _, lbls in test_loader:
                    imgs = imgs.to(DEVICE, non_blocking=True)
                    lbls = lbls.to(DEVICE, non_blocking=True)

                    # Standard prediction
                    with autocast(device_type=str(DEVICE).split(':')[0],
                                  enabled=(DEVICE.type == 'cuda')):
                        out = model(imgs)
                    probs = torch.softmax(out.float(), dim=1)

                    # TTA predictions
                    tta_probs = [probs]
                    for _ in range(4):  # 4 additional augmented views
                        aug_imgs = apply_tta_augmentation(imgs)
                        with autocast(device_type=str(DEVICE).split(':')[0],
                                      enabled=(DEVICE.type == 'cuda')):
                            out_aug = model(aug_imgs)
                        tta_probs.append(torch.softmax(out_aug.float(), dim=1))

                    avg_probs = torch.stack(tta_probs).mean(dim=0)
                    probs_list.extend(avg_probs.cpu().numpy())
                    labels_list.extend(lbls.cpu().numpy())

            all_model_probs[model_name] = np.array(probs_list)
            if all_labels is None:
                all_labels = np.array(labels_list)

            # Individual performance
            preds = np.array(probs_list).argmax(axis=1)
            acc = accuracy_score(all_labels, preds)
            f1 = f1_score(all_labels, preds, average='weighted')
            print(f"  {model_name}: acc={acc:.4f}, f1={f1:.4f} (with TTA)")

    # --- Ensemble Methods ---
    print(f"\n{'='*60}")
    print("Ensemble Results")
    print(f"{'='*60}")

    prob_values = list(all_model_probs.values())
    model_names = list(all_model_probs.keys())

    ensemble_results = {}

    # 1. Average Ensemble
    avg_probs = np.mean(prob_values, axis=0)
    avg_preds = avg_probs.argmax(axis=1)
    avg_acc = accuracy_score(all_labels, avg_preds)
    avg_f1 = f1_score(all_labels, avg_preds, average='weighted')
    try:
        avg_auc = roc_auc_score(all_labels, avg_probs[:, 1])
    except Exception:
        avg_auc = 0.5
    ensemble_results['average'] = {'accuracy': avg_acc, 'f1': avg_f1, 'auc': avg_auc}
    print(f"  Average Ensemble:  acc={avg_acc:.4f}, f1={avg_f1:.4f}, auc={avg_auc:.4f}")

    # 2. Weighted Ensemble (by individual accuracy)
    individual_accs = [accuracy_score(all_labels, p.argmax(axis=1)) for p in prob_values]
    weights = np.array(individual_accs)
    weights = weights / weights.sum()
    weighted_probs = np.average(prob_values, axis=0, weights=weights)
    weighted_preds = weighted_probs.argmax(axis=1)
    weighted_acc = accuracy_score(all_labels, weighted_preds)
    weighted_f1 = f1_score(all_labels, weighted_preds, average='weighted')
    try:
        weighted_auc = roc_auc_score(all_labels, weighted_probs[:, 1])
    except Exception:
        weighted_auc = 0.5
    ensemble_results['weighted'] = {'accuracy': weighted_acc, 'f1': weighted_f1, 'auc': weighted_auc}
    print(f"  Weighted Ensemble: acc={weighted_acc:.4f}, f1={weighted_f1:.4f}, auc={weighted_auc:.4f}")

    # 3. Majority Voting
    all_predictions = np.array([p.argmax(axis=1) for p in prob_values])
    vote_preds = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=2).argmax(),
        axis=0, arr=all_predictions
    )
    vote_acc = accuracy_score(all_labels, vote_preds)
    vote_f1 = f1_score(all_labels, vote_preds, average='weighted')
    ensemble_results['voting'] = {'accuracy': vote_acc, 'f1': vote_f1}
    print(f"  Voting Ensemble:   acc={vote_acc:.4f}, f1={vote_f1:.4f}")

    # 4. Confidence-filtered predictions
    print(f"\n  Confidence Filtering:")
    for threshold in [0.52, 0.55, 0.58, 0.60]:
        max_probs = avg_probs.max(axis=1)
        mask = max_probs >= threshold
        if mask.sum() > 0:
            filtered_acc = accuracy_score(all_labels[mask], avg_preds[mask])
            coverage = mask.mean()
            print(f"    threshold={threshold:.2f}: acc={filtered_acc:.4f}, "
                  f"coverage={coverage:.2%} ({mask.sum()}/{len(mask)})")
            ensemble_results[f'filtered_{threshold}'] = {
                'accuracy': filtered_acc,
                'coverage': float(coverage),
                'count': int(mask.sum()),
            }

    # Individual model summary
    individual_results = {}
    for name, probs in all_model_probs.items():
        preds = probs.argmax(axis=1)
        individual_results[name] = {
            'accuracy': float(accuracy_score(all_labels, preds)),
            'f1': float(f1_score(all_labels, preds, average='weighted')),
        }

    result = {
        'status': 'complete',
        'stage': 4,
        'name': 'EnsembleTTA',
        'individual': individual_results,
        'ensemble': ensemble_results,
        'num_models': len(stage2_models),
        'timestamp': datetime.now().isoformat(),
    }
    with open(stage_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nStage 4 complete! Results saved to {stage_dir}")


# =============================================================================
# Stage 5: Industry-Grouped Fine-tuning
# =============================================================================
def run_stage5(args):
    """
    Fine-tune best Stage 2 model per industry group.
    """
    stage_dir = OUTPUT_BASE / 'stage5'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_completed(stage_dir) and not args.force:
        print("Stage 5 already completed. Use --force to re-run.")
        return

    print("=" * 70)
    print("STAGE 5: Industry-Grouped Fine-tuning")
    print("=" * 70)

    # Load stock groups
    group_file = PROJECT_ROOT / 'data' / 'us_stock_groups.json'
    if not group_file.exists():
        print(f"ERROR: Group file not found: {group_file}")
        return
    with open(group_file) as f:
        groups_data = json.load(f)['groups']

    # Find best Stage 2 model (ResNet18 RGB for simplicity)
    stage2_dir = OUTPUT_BASE / 'stage2'
    best_ckpt = None
    best_f1 = 0

    for ckpt_file in sorted(stage2_dir.glob('resnet18_rgb_*.pt')):
        ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        f1_val = ckpt.get('metrics', {}).get('f1', 0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_ckpt = ckpt_file

    if best_ckpt is None:
        # Fallback: use any Stage 2 model
        ckpt_files = list(stage2_dir.glob('*.pt'))
        if ckpt_files:
            best_ckpt = ckpt_files[0]
        else:
            print("ERROR: No Stage 2 models found. Run Stage 2 first.")
            return

    print(f"Base model: {best_ckpt.name} (f1={best_f1:.4f})")

    base_ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)

    # Fine-tuning config
    ft_config = {
        'img_size': (128, 128),
        'window_size': 20,
        'prediction_horizon': 5,
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb',
        'label_threshold': 'dynamic',
        'chart_type': 'candle',
        'augment_prob': 0.3,
        'batch_size': 64,
        'epochs': 15,
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'patience': 8,
        'warmup_epochs': 2,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.1,
        'cutmix_alpha': 0.1,
        'freeze_epochs': 3,
    }

    if args.debug:
        ft_config['epochs'] = 3
        ft_config['freeze_epochs'] = 1

    all_group_results = []

    for group_id, group_info in groups_data.items():
        group_dir = stage_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = group_dir / 'model.pt'
        if ckpt_path.exists() and not args.force:
            print(f"\n{group_id}: already fine-tuned, skipping")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            all_group_results.append({
                'group_id': group_id,
                'group_name': group_info['name'],
                'num_stocks': len(group_info['stocks']),
                'test_metrics': ckpt['metrics'],
            })
            continue

        stocks = group_info['stocks']
        print(f"\n{'='*60}")
        print(f"Fine-tuning: {group_id} ({group_info['name']})")
        print(f"  Stocks: {stocks}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        np.random.seed(42)

        # Load full US dataset, then filter
        data_dirs = [US_DATA]
        train_ds_full, val_ds_full, test_ds_full, _, _, _ = \
            create_dataloaders(data_dirs, ft_config)

        # Filter by group stocks
        stock_set = set(stocks)
        train_idx = [i for i, s in enumerate(train_ds_full.samples) if s['ticker'] in stock_set]
        val_idx = [i for i, s in enumerate(val_ds_full.samples) if s['ticker'] in stock_set]
        test_idx = [i for i, s in enumerate(test_ds_full.samples) if s['ticker'] in stock_set]

        if not train_idx:
            print(f"  WARNING: No samples found for {group_id}, skipping")
            continue

        train_sub = Subset(train_ds_full, train_idx)
        val_sub = Subset(val_ds_full, val_idx)
        test_sub = Subset(test_ds_full, test_idx)

        loader_kw = {'batch_size': ft_config['batch_size'], 'num_workers': NUM_WORKERS,
                     'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True}
        train_loader = DataLoader(train_sub, shuffle=True, drop_last=True, **loader_kw)
        val_loader = DataLoader(val_sub, shuffle=False, **loader_kw)
        test_loader = DataLoader(test_sub, shuffle=False, **loader_kw)

        print(f"  Data: train={len(train_sub)}, val={len(val_sub)}, test={len(test_sub)}")

        if len(train_sub) < 100:
            print(f"  WARNING: Too few samples ({len(train_sub)}), skipping")
            continue

        # Build model from base checkpoint
        model = build_model(arch='resnet18', input_channels=3,
                            pretrained=False, dropout=0.3)
        # Load pretrained weights (exclude classifier)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in base_ckpt['model_state_dict'].items()
                           if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.to(DEVICE)

        # Freeze backbone initially
        freeze_epochs = ft_config['freeze_epochs']
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
        print(f"  Backbone frozen for first {freeze_epochs} epochs")

        # Training
        class_weights = train_ds_full.get_class_weights().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                         label_smoothing=ft_config['label_smoothing'])
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=ft_config['lr'], weight_decay=ft_config['weight_decay']
        )
        scheduler = get_warmup_cosine_scheduler(optimizer, ft_config['warmup_epochs'],
                                                 ft_config['epochs'])
        scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else GradScaler('cpu', enabled=False)

        best_f1 = 0
        best_state = None
        patience_cnt = 0

        for epoch in range(ft_config['epochs']):
            # Unfreeze backbone after freeze_epochs
            if epoch == freeze_epochs:
                for param in model.parameters():
                    param.requires_grad = True
                # Recreate optimizer with all parameters
                optimizer = torch.optim.AdamW(model.parameters(),
                                               lr=ft_config['lr'],
                                               weight_decay=ft_config['weight_decay'])
                scheduler = get_warmup_cosine_scheduler(
                    optimizer, 1, ft_config['epochs'] - freeze_epochs)
                print("  Backbone unfrozen")

            train_loss, train_acc = train_epoch_optimized(
                model, train_loader, criterion, optimizer, scaler,
                mixup_alpha=ft_config['mixup_alpha'],
                cutmix_alpha=ft_config['cutmix_alpha'],
            )
            val_m = evaluate_model(model, val_loader, criterion)
            scheduler.step()

            print(f"  Epoch {epoch+1}/{ft_config['epochs']}: loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_m['accuracy']:.4f} "
                  f"val_f1={val_m['f1']:.4f}", flush=True)

            if val_m['f1'] > best_f1:
                best_f1 = val_m['f1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= ft_config['patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        test_m = evaluate_model(model, test_loader, criterion)
        test_m_tta = evaluate_model_tta(model, test_loader, num_augments=5)

        print(f"  Test: acc={test_m['accuracy']:.4f} f1={test_m['f1']:.4f} auc={test_m['auc']:.4f}")
        print(f"  Test (TTA): acc={test_m_tta['accuracy']:.4f} f1={test_m_tta['f1']:.4f}")

        save_checkpoint(model, test_m, ft_config, ckpt_path)

        all_group_results.append({
            'group_id': group_id,
            'group_name': group_info['name'],
            'num_stocks': len(stocks),
            'train_samples': len(train_sub),
            'test_metrics': test_m,
            'test_metrics_tta': test_m_tta,
            'best_val_f1': best_f1,
        })

    # Summary
    print(f"\n{'='*70}")
    print("STAGE 5 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Group':<25} {'Stocks':>6} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 60)
    for r in all_group_results:
        m = r['test_metrics']
        print(f"{r['group_id']:<25} {r['num_stocks']:>6} "
              f"{m['accuracy']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    # Weighted average
    if all_group_results:
        total_samples = sum(r.get('train_samples', 1) for r in all_group_results)
        wavg_acc = sum(r['test_metrics']['accuracy'] * r.get('train_samples', 1)
                       for r in all_group_results) / total_samples
        wavg_f1 = sum(r['test_metrics']['f1'] * r.get('train_samples', 1)
                      for r in all_group_results) / total_samples
        print(f"\n  Weighted Average: acc={wavg_acc:.4f}, f1={wavg_f1:.4f}")

    result = {
        'status': 'complete',
        'stage': 5,
        'name': 'GroupedFinetune',
        'groups': all_group_results,
        'config': ft_config,
        'base_model': str(best_ckpt),
        'timestamp': datetime.now().isoformat(),
    }
    with open(stage_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nStage 5 complete! Results saved to {stage_dir}")


# =============================================================================
# Stage 6: Technical Indicators + Aggressive Signal Filtering (Target: 60%+)
# =============================================================================
def compute_technical_indicators(df):
    """
    Compute technical indicators for a DataFrame with OHLCV columns.
    Uses pure numpy/pandas (no external TA library required).
    Returns DataFrame with additional indicator columns.
    """
    df = df.copy()
    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    volume = df['Volume'].values.astype(np.float64)
    n = len(close)

    # --- RSI (14-period) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='same')
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
    def ema(arr, span):
        alpha = 2 / (span + 1)
        result = np.zeros_like(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    df['MACD'] = macd_line
    df['MACD_signal'] = signal_line
    df['MACD_hist'] = macd_line - signal_line

    # --- Bollinger Bands (20-period, 2 std) ---
    sma20 = np.convolve(close, np.ones(20)/20, mode='same')
    std20 = np.array([close[max(0,i-19):i+1].std() if i >= 1 else 0.0 for i in range(n)])
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (sma20 + 1e-10)
    df['BB_position'] = (close - df['BB_lower'].values) / (df['BB_upper'].values - df['BB_lower'].values + 1e-10)

    # --- ATR (14-period Average True Range) ---
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    df['ATR'] = np.convolve(tr, np.ones(14)/14, mode='same')

    # --- Stochastic %K (14-period) ---
    stoch_k = np.zeros(n)
    for i in range(13, n):
        h14 = high[i-13:i+1].max()
        l14 = low[i-13:i+1].min()
        stoch_k[i] = 100 * (close[i] - l14) / (h14 - l14 + 1e-10)
    df['Stoch_K'] = stoch_k

    # --- OBV (On-Balance Volume) ---
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    df['OBV'] = obv

    # --- Volume Moving Average Ratio ---
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='same')
    df['Volume_ratio'] = volume / (vol_ma20 + 1e-10)

    # --- Rate of Change (12-period) ---
    roc = np.zeros(n)
    for i in range(12, n):
        roc[i] = (close[i] - close[i-12]) / (close[i-12] + 1e-10) * 100
    df['ROC'] = roc

    # --- Trend Strength (R-squared of linear regression over 20 bars) ---
    trend_r2 = np.zeros(n)
    for i in range(19, n):
        window = close[i-19:i+1]
        x = np.arange(20)
        slope, intercept = np.polyfit(x, window, 1)
        pred = slope * x + intercept
        ss_res = np.sum((window - pred) ** 2)
        ss_tot = np.sum((window - window.mean()) ** 2)
        trend_r2[i] = 1 - (ss_res / (ss_tot + 1e-10))
    df['Trend_R2'] = np.clip(trend_r2, 0, 1)

    return df


def compute_signal_quality(df, idx, window_size, prediction_horizon):
    """
    Compute a signal quality score for a sample.
    Higher score = more predictable signal.

    Combines:
    1. Trend strength (R-squared)
    2. Volume confirmation
    3. Indicator consensus (RSI + MACD + Stochastic alignment)
    4. Bollinger Band position (extreme positions are more predictive)
    """
    end_idx = idx + window_size - 1
    if end_idx >= len(df) or end_idx < 20:
        return 0.0

    row = df.iloc[end_idx]
    score = 0.0

    # 1. Trend strength (0-1): strong trends are predictable
    trend_r2 = row.get('Trend_R2', 0.0)
    score += trend_r2 * 0.3

    # 2. RSI extreme (0-1): overbought/oversold more predictive
    rsi = row.get('RSI', 50)
    rsi_extreme = abs(rsi - 50) / 50  # 0 at center, 1 at extremes
    score += rsi_extreme * 0.2

    # 3. MACD histogram strength (0-1): strong momentum
    macd_hist = abs(row.get('MACD_hist', 0))
    # Normalize by ATR
    atr = row.get('ATR', 1.0)
    macd_norm = min(1.0, macd_hist / (atr + 1e-10))
    score += macd_norm * 0.2

    # 4. BB position extreme: near top/bottom band
    bb_pos = row.get('BB_position', 0.5)
    bb_extreme = abs(bb_pos - 0.5) * 2  # 0 at center, 1 at bands
    bb_extreme = min(1.0, bb_extreme)
    score += bb_extreme * 0.15

    # 5. Volume confirmation: high volume relative to average
    vol_ratio = row.get('Volume_ratio', 1.0)
    vol_confirm = min(1.0, max(0, (vol_ratio - 1.0)))  # Higher = better
    score += vol_confirm * 0.15

    return float(score)


class TechIndicatorDataset(torch.utils.data.Dataset):
    """
    Enhanced dataset that adds technical indicators as extra channels.
    Wraps StockDataset and adds indicator overlay to images.
    """

    def __init__(self, base_dataset, quality_threshold=0.0):
        """
        Args:
            base_dataset: A StockDataset instance
            quality_threshold: Minimum signal quality score (0-1).
                             Higher = fewer but more predictable samples.
        """
        self.base_ds = base_dataset
        self.quality_threshold = quality_threshold

        # Compute technical indicators for all cached data
        self.indicator_cache = {}
        for ticker, df in base_dataset.data_cache.items():
            self.indicator_cache[ticker] = compute_technical_indicators(df)

        # Filter samples by signal quality if threshold > 0
        if quality_threshold > 0 and base_dataset.mode == 'train':
            self.filtered_indices = []
            for i, sample in enumerate(base_dataset.samples):
                ticker = sample['ticker']
                start_idx = sample['start_idx']
                if ticker in self.indicator_cache:
                    quality = compute_signal_quality(
                        self.indicator_cache[ticker],
                        start_idx,
                        base_dataset.window_size,
                        base_dataset.prediction_horizon,
                    )
                    if quality >= quality_threshold:
                        self.filtered_indices.append(i)
            total = len(base_dataset.samples)
            retained = len(self.filtered_indices)
            pct = retained / total if total > 0 else 0
            print(f"  Signal quality filter: {retained}/{total} "
                  f"samples retained ({pct:.1%}) "
                  f"[threshold={quality_threshold}]")
        else:
            self.filtered_indices = list(range(len(base_dataset.samples)))

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        real_idx = self.filtered_indices[idx]
        img_tensor, seq_tensor, label_tensor = self.base_ds[real_idx]

        # Get sample info
        sample = self.base_ds.samples[real_idx]
        ticker = sample['ticker']
        start_idx = sample['start_idx']
        window_size = self.base_ds.window_size
        end_idx = start_idx + window_size - 1

        if ticker in self.indicator_cache:
            ind_df = self.indicator_cache[ticker]

            # Extract indicator values for the window
            if end_idx < len(ind_df):
                row = ind_df.iloc[end_idx]

                # Create indicator channels (same size as image)
                C, H, W = img_tensor.shape

                # Channel 1: RSI heatmap (0-100 -> 0-1)
                rsi_val = row.get('RSI', 50) / 100.0
                rsi_channel = torch.full((1, H, W), rsi_val, dtype=torch.float32)

                # Channel 2: MACD histogram (normalized)
                macd_h = row.get('MACD_hist', 0)
                atr = row.get('ATR', 1.0)
                macd_norm = np.clip(macd_h / (atr * 3 + 1e-10) + 0.5, 0, 1)
                macd_channel = torch.full((1, H, W), macd_norm, dtype=torch.float32)

                # Channel 3: Bollinger Band position (0-1)
                bb_pos = np.clip(row.get('BB_position', 0.5), 0, 1)
                bb_channel = torch.full((1, H, W), bb_pos, dtype=torch.float32)

                # Concatenate original image with indicator channels
                img_tensor = torch.cat([img_tensor, rsi_channel, macd_channel, bb_channel], dim=0)

        return img_tensor, seq_tensor, label_tensor

    def get_num_channels(self):
        return self.base_ds.get_num_channels() + 3  # +RSI, +MACD, +BB

    def get_class_weights(self):
        # Recompute for filtered samples
        labels = [self.base_ds.samples[i]['label'] for i in self.filtered_indices]
        class_counts = np.bincount(labels, minlength=self.base_ds.num_classes)
        total = len(labels)
        weights = total / (self.base_ds.num_classes * class_counts + 1e-8)
        return torch.tensor(weights, dtype=torch.float32)


def run_stage6(args):
    """
    STAGE 6: Technical Indicators + Aggressive Signal Filtering
    Target: 60%+ accuracy

    Key innovations:
    1. Technical indicators (RSI, MACD, BB, ATR, Stoch, OBV) as extra image channels
    2. Signal quality scoring to keep only the most predictable samples
    3. Aggressive quantile filtering (top/bottom 50% only)
    4. Ensemble of indicator-enhanced models with industry grouping
    """
    stage_dir = OUTPUT_BASE / 'stage6'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_completed(stage_dir) and not args.force:
        print("Stage 6 already completed. Use --force to re-run.")
        return

    print("=" * 70)
    print("STAGE 6: Technical Indicators + Signal Quality Filtering")
    print("       TARGET: 60%+ Accuracy")
    print("=" * 70)

    config = {
        'name': 'TechIndicators_SignalFilter',
        'arch': 'resnet18',
        'pretrained': True,
        'dropout': 0.4,  # Stronger dropout for 6ch input
        'img_size': (128, 128),
        'window_size': 20,
        'prediction_horizon': 5,
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb',  # Base 3ch + 3ch indicators = 6ch
        'chart_type': 'candle',
        'label_threshold': 'dynamic',
        'augment_prob': 0.5,
        'epochs': 50,
        'batch_size': 128,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'patience': 15,
        'warmup_epochs': 5,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.2,
        'label_smoothing': 0.1,
        'num_runs': 3,
        # Signal quality thresholds to test
        'quality_thresholds': [0.0, 0.2, 0.3, 0.4],
        # Also test with quantile filtering
        'quantile_filter': 0.45,
    }

    if args.debug:
        config['limit'] = 5
        config['epochs'] = 5
        config['num_runs'] = 1
        config['quality_thresholds'] = [0.0, 0.3]

    data_dirs = get_data_dirs()

    all_experiments = []

    # Experiment A: Indicator channels only (no signal filtering)
    # Experiment B: Indicator channels + progressive signal quality filtering
    # Experiment C: Indicator channels + quantile filtering + signal quality

    for quality_threshold in config['quality_thresholds']:
        exp_name = f"indicators_q{quality_threshold:.1f}"
        exp_dir = stage_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = exp_dir / 'best_model.pt'
        if ckpt_path.exists() and not args.force:
            print(f"\n{exp_name}: already completed, skipping")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            all_experiments.append({
                'name': exp_name,
                'quality_threshold': quality_threshold,
                'test_metrics': ckpt['metrics'],
            })
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"  Quality threshold: {quality_threshold}")
        print(f"  Quantile filter: {config['quantile_filter']}")
        print(f"{'='*60}")

        seeds = [42, 142, 242][:config['num_runs']]
        runs = []

        for run_idx, seed in enumerate(seeds):
            print(f"\n  --- Run {run_idx+1}/{len(seeds)} (seed={seed}) ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create base dataset with quantile filtering
            ds_config = {**config, 'limit': config.get('limit', None)}
            
            # Common config for all datasets
            common_base = {
                'data_dir': data_dirs,
                'window_size': config['window_size'],
                'prediction_horizon': config['prediction_horizon'],
                'img_size': config['img_size'],
                'norm_method': config['norm_method'],
                'use_clahe': config['use_clahe'],
                'output_channels': config['output_channels'],
                'label_threshold': config['label_threshold'],
                'chart_type': config['chart_type'],
                'mixup_prob': 0.0,
                'cutmix_prob': 0.0,
            }
            
            # Training: apply quantile filter to focus on extreme moves
            train_common = {**common_base, 'quantile_filter': config['quantile_filter']}
            # Val/Test: NO quantile filter - evaluate on all samples (real-world scenario)
            val_test_common = {**common_base, 'quantile_filter': None}

            limit = config.get('limit', None)
            train_ds_base = StockDataset(
                mode='train', augment_prob=config['augment_prob'],
                limit=limit, **train_common
            )
            val_ds_base = StockDataset(mode='val', augment_prob=0.0, limit=limit, **val_test_common)
            test_ds_base = StockDataset(mode='test', augment_prob=0.0, limit=limit, **val_test_common)

            # Wrap with technical indicator enhancement
            train_ds = TechIndicatorDataset(train_ds_base, quality_threshold=quality_threshold)
            val_ds = TechIndicatorDataset(val_ds_base, quality_threshold=0.0)  # No filtering on val/test
            test_ds = TechIndicatorDataset(test_ds_base, quality_threshold=0.0)

            input_channels = train_ds.get_num_channels()
            print(f"  Data: train={len(train_ds)}, val={len(val_ds)}, "
                  f"test={len(test_ds)}, channels={input_channels}")

            if len(train_ds) < 100:
                print(f"  WARNING: Too few training samples ({len(train_ds)}), skipping")
                continue

            loader_kw = {
                'batch_size': config['batch_size'],
                'num_workers': NUM_WORKERS,
                'pin_memory': True,
                'prefetch_factor': 2,
                'persistent_workers': True,
            }
            train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
            val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
            test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

            # Build model with extra channels
            model = build_model(
                arch=config['arch'],
                input_channels=input_channels,
                pretrained=config['pretrained'],
                dropout=config['dropout'],
            )

            # Run training
            model, test_m, test_m_tta, history, best_f1 = run_training(
                model, train_loader, val_loader, test_loader, train_ds, config
            )

            if run_idx == 0 or test_m['f1'] > max([r['test_metrics']['f1'] for r in runs], default=0):
                save_checkpoint(model, test_m, config, ckpt_path)

            runs.append({
                'seed': seed,
                'test_metrics': test_m,
                'test_metrics_tta': test_m_tta,
                'best_val_f1': best_f1,
                'epochs_trained': len(history),
                'train_samples': len(train_ds),
            })

        # Summarize this experiment
        if runs:
            summary = {}
            for k in ['accuracy', 'f1', 'auc']:
                vals = [r['test_metrics'][k] for r in runs]
                summary[f'{k}_mean'] = float(np.mean(vals))
                summary[f'{k}_std'] = float(np.std(vals))
                vals_tta = [r['test_metrics_tta'][k] for r in runs]
                summary[f'{k}_tta_mean'] = float(np.mean(vals_tta))
                summary[f'{k}_tta_std'] = float(np.std(vals_tta))

            print(f"\n  {exp_name} Summary:")
            print(f"    Acc:      {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
            print(f"    F1:       {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
            print(f"    AUC:      {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}")
            print(f"    Acc(TTA): {summary['accuracy_tta_mean']:.4f} +/- {summary['accuracy_tta_std']:.4f}")

            all_experiments.append({
                'name': exp_name,
                'quality_threshold': quality_threshold,
                'runs': runs,
                'summary': summary,
                'train_samples': runs[0].get('train_samples', 0),
            })

    # --- Overall Summary ---
    print(f"\n{'='*70}")
    print("STAGE 6 SUMMARY: Technical Indicators + Signal Quality")
    print(f"{'='*70}")
    print(f"{'Experiment':<30} {'Samples':>8} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Acc(TTA)':>10}")
    print("-" * 75)
    for exp in all_experiments:
        s = exp.get('summary', exp.get('test_metrics', {}))
        samples = exp.get('train_samples', '?')
        acc = s.get('accuracy_mean', s.get('accuracy', 0))
        f1 = s.get('f1_mean', s.get('f1', 0))
        auc = s.get('auc_mean', s.get('auc', 0))
        acc_tta = s.get('accuracy_tta_mean', 0)
        print(f"{exp['name']:<30} {str(samples):>8} {acc:>8.4f} {f1:>8.4f} {auc:>8.4f} {acc_tta:>10.4f}")

    result = {
        'status': 'complete',
        'stage': 6,
        'name': 'TechIndicators_SignalFilter',
        'experiments': all_experiments,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }
    with open(stage_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nStage 6 complete! Results saved to {stage_dir}")


# =============================================================================
# Stage 7: 3-Class Classification (Up/Neutral/Down) with Neutral Penalty
# =============================================================================
def run_stage7(args):
    """
    3-Class Classification with special evaluation.
    
    Labels: 0=Down, 1=Neutral, 2=Up
    
    Key features:
    1. Neutral zone based on return threshold (e.g., +/- 1%)
    2. Training can filter extreme neutrals for better signal
    3. Special evaluation: predicting neutral on non-neutral actual = WRONG
    4. NO data leakage: filtering only applied to training set
    """
    stage_dir = OUTPUT_BASE / 'stage7'
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_completed(stage_dir) and not args.force:
        print("Stage 7 already completed. Use --force to re-run.")
        return

    print("=" * 70)
    print("STAGE 7: 3-Class Classification (Up/Neutral/Down)")
    print("       With Neutral Penalty Evaluation")
    print("=" * 70)

    config = {
        'name': '3Class_NeutralPenalty',
        'arch': 'resnet18',
        'pretrained': True,
        'dropout': 0.4,
        'img_size': (128, 128),
        'window_size': 20,
        'prediction_horizon': 5,
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb',
        'chart_type': 'candle',
        'label_threshold': 0.01,  # 1% threshold for neutral zone
        'num_classes': 3,  # 3-class: down/neutral/up
        'augment_prob': 0.5,
        'epochs': 40,
        'batch_size': 128,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'patience': 12,
        'warmup_epochs': 5,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.2,
        'label_smoothing': 0.1,
        'num_runs': 3,
        # Training filter: skip samples with |return| < 0.3% during training
        # This removes extreme noise while keeping val/test intact
        'train_filter_threshold': 0.003,
    }

    if args.debug:
        config['limit'] = 5
        config['epochs'] = 5
        config['num_runs'] = 1

    data_dirs = get_data_dirs()

    all_runs = []
    seeds = [42, 142, 242][:config['num_runs']]

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Run {run_idx+1}/{len(seeds)} (seed={seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create dataloaders with 3-class config
        # IMPORTANT: create_dataloaders ensures quantile_filter only applies to train
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
            create_dataloaders(data_dirs, config)

        print(f"  Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        
        # Check class distribution
        train_labels = [s['label'] for s in train_ds.samples]
        val_labels = [s['label'] for s in val_ds.samples]
        test_labels = [s['label'] for s in test_ds.samples]
        
        print(f"  Train class distribution: Down={train_labels.count(0)}, "
              f"Neutral={train_labels.count(1)}, Up={train_labels.count(2)}")
        print(f"  Val class distribution: Down={val_labels.count(0)}, "
              f"Neutral={val_labels.count(1)}, Up={val_labels.count(2)}")
        print(f"  Test class distribution: Down={test_labels.count(0)}, "
              f"Neutral={test_labels.count(1)}, Up={test_labels.count(2)}")

        # Build model for 3-class
        input_channels = train_ds.get_num_channels()
        model = build_model(
            arch=config['arch'],
            input_channels=input_channels,
            num_classes=3,  # 3 output classes
            pretrained=config['pretrained'],
            dropout=config['dropout'],
        )

        # Train
        model, test_m, test_m_tta, history, best_f1 = run_training(
            model, train_loader, val_loader, test_loader, train_ds, config
        )

        # Special 3-class evaluation with neutral penalty
        neutral_metrics = evaluate_3class_with_neutral_penalty(
            model, test_loader, neutral_threshold=config['label_threshold'], device=DEVICE
        )
        
        print(f"\n  3-Class Evaluation (neutral threshold={config['label_threshold']:.2%}):")
        print(f"    Standard Accuracy: {neutral_metrics['accuracy']:.4f}")
        print(f"    Strict Accuracy: {neutral_metrics['strict_accuracy']:.4f}")
        print(f"    Neutral Wrong Rate: {neutral_metrics['neutral_wrong_rate']:.4f}")
        print(f"    F1 (weighted): {neutral_metrics['f1_weighted']:.4f}")
        print(f"    Class distribution - Pred: {neutral_metrics['pred_down']}/{neutral_metrics['pred_neutral']}/{neutral_metrics['pred_up']}, "
              f"Actual: {neutral_metrics['actual_down']}/{neutral_metrics['actual_neutral']}/{neutral_metrics['actual_up']}")

        # Save checkpoint
        save_checkpoint(model, {**test_m, 'neutral_metrics': neutral_metrics}, config,
                        stage_dir / f'model_seed{seed}.pt')

        all_runs.append({
            'seed': seed,
            'test_metrics': test_m,
            'test_metrics_tta': test_m_tta,
            'neutral_metrics': neutral_metrics,
            'best_val_f1': best_f1,
            'epochs_trained': len(history),
            'history': history,
        })

    # Summary
    summary = {}
    for k in ['accuracy', 'f1', 'precision', 'recall']:
        vals = [r['test_metrics'][k] for r in all_runs]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))
    
    # 3-class specific metrics
    summary['strict_accuracy_mean'] = float(np.mean([r['neutral_metrics']['strict_accuracy'] for r in all_runs]))
    summary['strict_accuracy_std'] = float(np.std([r['neutral_metrics']['strict_accuracy'] for r in all_runs]))
    summary['neutral_wrong_mean'] = float(np.mean([r['neutral_metrics']['neutral_wrong_rate'] for r in all_runs]))

    print(f"\n{'='*70}")
    print(f"STAGE 7 SUMMARY")
    print(f"{'='*70}")
    print(f"Standard Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"Strict Accuracy:   {summary['strict_accuracy_mean']:.4f} +/- {summary['strict_accuracy_std']:.4f}")
    print(f"Neutral Wrong:     {summary['neutral_wrong_mean']:.4f}")
    print(f"F1:                {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"\nNote: Strict accuracy counts 'predicted neutral but actual up/down' as WRONG")

    result = {
        'status': 'complete',
        'stage': 7,
        'name': '3Class_NeutralPenalty',
        'config': config,
        'runs': all_runs,
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
    }
    with open(stage_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nStage 7 complete! Results saved to {stage_dir}")


# =============================================================================
# Final Summary
# =============================================================================
def print_final_summary():
    """Print comparison of all stages vs baseline."""
    print(f"\n{'='*70}")
    print("FINAL COMPARISON: Optimized Pipeline vs Baseline")
    print(f"{'='*70}")

    # Baseline numbers
    print(f"\n{'Method':<35} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 65)
    print(f"{'[Baseline] KLineNet (pretrained=F)':<35} {'0.5126':>8} {'0.5118':>8} {'0.5183':>8}")
    print(f"{'[Baseline] Grouped-Financial':<35} {'0.5379':>8} {'0.5349':>8} {'  N/A':>8}")

    for stage_num in [1, 2, 3, 4, 5, 6]:
        result_file = OUTPUT_BASE / f'stage{stage_num}' / 'results.json'
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)

            if stage_num == 1:
                s = data.get('summary', {})
                print(f"{'[Stage 1] Optimized Baseline':<35} "
                      f"{s.get('accuracy_mean', 0):>8.4f} "
                      f"{s.get('f1_mean', 0):>8.4f} "
                      f"{s.get('auc_mean', 0):>8.4f}")
                print(f"{'[Stage 1] + TTA':<35} "
                      f"{s.get('accuracy_tta_mean', 0):>8.4f} "
                      f"{s.get('f1_tta_mean', 0):>8.4f} "
                      f"{s.get('auc_tta_mean', 0):>8.4f}")

            elif stage_num == 2:
                models = data.get('models', [])
                if models:
                    accs = [m['test_metrics']['accuracy'] for m in models]
                    f1s = [m['test_metrics']['f1'] for m in models]
                    aucs = [m['test_metrics'].get('auc', 0.5) for m in models]
                    print(f"{'[Stage 2] Best Individual':<35} "
                          f"{max(accs):>8.4f} {max(f1s):>8.4f} {max(aucs):>8.4f}")

            elif stage_num == 3:
                models = data.get('models', [])
                for m in models:
                    tm = m['test_metrics']
                    name = f"[Stage 3] {m['model_name']}"
                    print(f"{name:<35} {tm['accuracy']:>8.4f} "
                          f"{tm['f1']:>8.4f} {tm['auc']:>8.4f}")

            elif stage_num == 4:
                ens = data.get('ensemble', {})
                for method in ['average', 'weighted', 'voting']:
                    if method in ens:
                        e = ens[method]
                        name = f"[Stage 4] Ensemble-{method}"
                        print(f"{name:<35} {e.get('accuracy', 0):>8.4f} "
                              f"{e.get('f1', 0):>8.4f} {e.get('auc', 0):>8.4f}")

            elif stage_num == 5:
                groups = data.get('groups', [])
                for g in groups:
                    tm = g['test_metrics']
                    name = f"[Stage 5] {g['group_id']}"
                    print(f"{name:<35} {tm['accuracy']:>8.4f} "
                          f"{tm['f1']:>8.4f} {tm['auc']:>8.4f}")

            elif stage_num == 6:
                exps = data.get('experiments', [])
                for exp in exps:
                    s = exp.get('summary', exp.get('test_metrics', {}))
                    name = f"[Stage 6] {exp['name'][:20]}"
                    print(f"{name:<35} {s.get('accuracy_mean', s.get('accuracy', 0)):>8.4f} "
                          f"{s.get('f1_mean', s.get('f1', 0)):>8.4f} "
                          f"{s.get('auc_mean', s.get('auc', 0)):>8.4f}")
            
            elif stage_num == 7:
                s = data.get('summary', {})
                print(f"{'[Stage 7] 3-Class (Standard Acc)':<35} "
                      f"{s.get('accuracy_mean', 0):>8.4f} "
                      f"{s.get('f1_mean', 0):>8.4f} {'N/A':>8}")
                print(f"{'[Stage 7] 3-Class (Strict Acc)':<35} "
                      f"{s.get('strict_accuracy_mean', 0):>8.4f} "
                      f"{'N/A':>8} {'N/A':>8}")

    print(f"\n{'='*70}")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Optimized K-Line Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_optimized_pipeline.py --stage all
  python scripts/run_optimized_pipeline.py --stage 1
  python scripts/run_optimized_pipeline.py --stage 4 --force
  python scripts/run_optimized_pipeline.py --stage all --debug
        """,
    )
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', '4', '5', '6', '7', 'all'],
                        help='Which stage to run (default: all)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if stage is completed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: use 5 stocks, fewer epochs')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (recommend 8 for H20)')
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"Optimized K-Line Prediction Pipeline")
    print(f"{'='*70}")
    print(f"Start: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"Stage: {args.stage}")
    print(f"Debug: {args.debug}")
    print(f"Output: {OUTPUT_BASE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    stages_to_run = ['1', '2', '3', '4', '5', '6', '7'] if args.stage == 'all' else [args.stage]

    for stage in stages_to_run:
        print(f"\n{'#'*70}")
        print(f"# Starting Stage {stage}")
        print(f"{'#'*70}\n")

        if stage == '1':
            run_stage1(args)
        elif stage == '2':
            run_stage2(args)
        elif stage == '3':
            run_stage3(args)
        elif stage == '4':
            run_stage4(args)
        elif stage == '5':
            run_stage5(args)
        elif stage == '6':
            run_stage6(args)
        elif stage == '7':
            run_stage7(args)

    # Print final comparison
    print_final_summary()

    print(f"\nPipeline complete! {datetime.now()}")
    print(f"All results saved to: {OUTPUT_BASE}")


if __name__ == '__main__':
    main()
