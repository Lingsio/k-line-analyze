"""
Transfer Learning Training Script

Implements two-stage training:
1. Pre-training on large universal dataset (all stocks)
2. Fine-tuning on target stocks or markets

Based on Xiu et al. (2021) finding that transfer learning boosts
Sharpe ratios by 0.4 on average in international markets.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from datetime import datetime
from tqdm import tqdm

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel
from core.config import settings


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    use_amp = scaler is not None
    
    for batch in tqdm(loader, desc="Training", leave=False):
        imgs, seqs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            imgs, seqs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def pretrain_universal(
    markets=['us', 'cn'],
    epochs=30,
    batch_size=128,
    lr=1e-3,
    device='cuda',
    output_name='universal_pretrained'
):
    """
    Stage 1: Pre-train on all available stocks.
    
    Creates a universal model that learns general K-line patterns
    across different markets and stocks.
    """
    print("="*70)
    print("Stage 1: Universal Pre-training")
    print("="*70)
    
    # Collect data from all markets
    data_dirs = []
    for market in markets:
        data_dir = PROJECT_ROOT / 'data' / 'raw' / market
        if data_dir.exists():
            data_dirs.append(str(data_dir))
    
    print(f"Loading data from: {data_dirs}")
    
    # Create dataset with all stocks
    train_ds = StockDataset(
        data_dir=data_dirs,
        window_size=10,
        prediction_horizon=5,
        img_size=(256, 256),
        mode='train',
        chart_type='ohlc',
        augment_prob=0.3,
        label_threshold='dynamic',
    )
    
    val_ds = StockDataset(
        data_dir=data_dirs,
        window_size=10,
        prediction_horizon=5,
        img_size=(256, 256),
        mode='val',
        chart_type='ohlc',
        augment_prob=0.0,
        label_threshold='dynamic',
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = CNNModel(
        num_classes=2,
        input_channels=3,
        arch='resnet18',
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Training loop
    best_val_acc = 0
    history = []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save pre-trained weights
            output_dir = settings.MODELS_DIR / 'transfer_learning'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'markets': markets,
                    'window_size': 10,
                    'img_size': (256, 256),
                    'chart_type': 'ohlc',
                }
            }
            torch.save(checkpoint, output_dir / f'{output_name}.pt')
            print(f"  ** Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save history
    with open(output_dir / f'{output_name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nPre-training complete! Best val acc: {best_val_acc:.2f}%")
    return model, history


def finetune_target(
    pretrained_path,
    target_market='us',
    target_symbols=None,
    epochs=20,
    batch_size=64,
    lr=1e-4,  # Lower LR for fine-tuning
    freeze_backbone_epochs=5,
    device='cuda',
    output_name='finetuned'
):
    """
    Stage 2: Fine-tune on target stocks/market.
    
    Uses lower learning rate and optional backbone freezing
    to preserve general features while adapting to target.
    """
    print("="*70)
    print("Stage 2: Fine-tuning")
    print("="*70)
    
    # Load pre-trained model
    checkpoint = torch.load(pretrained_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Loaded pre-trained model from: {pretrained_path}")
    print(f"Pre-trained on: {config['markets']}")
    
    # Create target dataset
    if target_symbols:
        # Filter to specific symbols
        data_dirs = []
        for symbol in target_symbols:
            data_dir = PROJECT_ROOT / 'data' / 'raw' / target_market
            if (data_dir / f'{symbol}.csv').exists() or (data_dir / f'{symbol}.parquet').exists():
                data_dirs.append(str(data_dir))
    else:
        data_dirs = [str(PROJECT_ROOT / 'data' / 'raw' / target_market)]
    
    train_ds = StockDataset(
        data_dir=data_dirs,
        window_size=config.get('window_size', 10),
        prediction_horizon=5,
        img_size=config.get('img_size', (256, 256)),
        mode='train',
        chart_type=config.get('chart_type', 'ohlc'),
        augment_prob=0.3,
        label_threshold='dynamic',
    )
    
    val_ds = StockDataset(
        data_dir=data_dirs,
        window_size=config.get('window_size', 10),
        prediction_horizon=5,
        img_size=config.get('img_size', (256, 256)),
        mode='val',
        chart_type=config.get('chart_type', 'ohlc'),
        augment_prob=0.0,
        label_threshold='dynamic',
    )
    
    test_ds = StockDataset(
        data_dir=data_dirs,
        window_size=config.get('window_size', 10),
        prediction_horizon=5,
        img_size=config.get('img_size', (256, 256)),
        mode='test',
        chart_type=config.get('chart_type', 'ohlc'),
        augment_prob=0.0,
        label_threshold='dynamic',
    )
    
    print(f"Target: {target_market} ({target_symbols or 'all'})")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model and load pre-trained weights
    model = CNNModel(
        num_classes=2,
        input_channels=3,
        arch='resnet18',
        pretrained=False  # We'll load our own weights
    ).to(device)
    
    # Load pre-trained backbone (excluding classifier)
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    
    # Filter out classifier weights (different task may have different classes)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                       if k in model_dict and 'classifier' not in k and 'fc' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"Loaded {len(pretrained_dict)} layers from pre-trained model")
    
    # Loss and optimizer (lower learning rate)
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Optionally freeze backbone
    if freeze_backbone_epochs > 0:
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
        print(f"Backbone frozen for first {freeze_backbone_epochs} epochs")
    
    # Training loop
    best_val_acc = 0
    history = []
    
    print(f"\nFine-tuning for {epochs} epochs (lr={lr})...")
    for epoch in range(epochs):
        # Unfreeze backbone if needed
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            for param in model.parameters():
                param.requires_grad = True
            print("Backbone unfrozen")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = settings.MODELS_DIR / 'transfer_learning'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'pretrained_from': str(pretrained_path),
            }
            torch.save(checkpoint, output_dir / f'{output_name}.pt')
            print(f"  ** Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Final test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Save history
    with open(output_dir / f'{output_name}_history.json', 'w') as f:
        json.dump({'history': history, 'test_acc': test_acc}, f, indent=2)
    
    return model, history, test_acc


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Training')
    parser.add_argument('--stage', choices=['pretrain', 'finetune', 'both'], default='both')
    parser.add_argument('--markets', nargs='+', default=['us', 'cn'])
    parser.add_argument('--target-market', default='us')
    parser.add_argument('--target-symbols', nargs='+', default=None)
    parser.add_argument('--pretrained-path', default=None)
    parser.add_argument('--epochs-pretrain', type=int, default=30)
    parser.add_argument('--epochs-finetune', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr-pretrain', type=float, default=1e-3)
    parser.add_argument('--lr-finetune', type=float, default=1e-4)
    parser.add_argument('--freeze-epochs', type=int, default=5)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.stage in ['pretrain', 'both']:
        # Stage 1: Pre-training
        pretrained_path = settings.MODELS_DIR / 'transfer_learning' / 'universal_pretrained.pt'
        
        model, history = pretrain_universal(
            markets=args.markets,
            epochs=args.epochs_pretrain,
            batch_size=args.batch_size,
            lr=args.lr_pretrain,
            device=device,
            output_name='universal_pretrained'
        )
    else:
        pretrained_path = args.pretrained_path
    
    if args.stage in ['finetune', 'both']:
        # Stage 2: Fine-tuning
        if pretrained_path is None:
            pretrained_path = settings.MODELS_DIR / 'transfer_learning' / 'universal_pretrained.pt'
        
        model, history, test_acc = finetune_target(
            pretrained_path=pretrained_path,
            target_market=args.target_market,
            target_symbols=args.target_symbols,
            epochs=args.epochs_finetune,
            batch_size=args.batch_size // 2,  # Smaller batch for fine-tuning
            lr=args.lr_finetune,
            freeze_backbone_epochs=args.freeze_epochs,
            device=device,
            output_name=f'finetuned_{args.target_market}'
        )
    
    print("\nTransfer learning complete!")


if __name__ == '__main__':
    main()
