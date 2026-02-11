"""
Ensemble Training Script

Trains multiple diverse models and combines their predictions
for improved robustness and accuracy.

Strategies:
1. Different architectures (ResNet18, EfficientNet, etc.)
2. Different data representations (OHLC, GAF, Candle)
3. Different time scales (5d, 10d, 20d)
4. Bootstrap aggregation (different data subsets)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from datetime import datetime
from collections import defaultdict

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel
from src.models.multiscale_cnn import (
    MultiScaleKLineEncoder,
    HierarchicalMultiScaleCNN,
    LightweightMultiScaleCNN
)
from core.config import settings


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple CNN models with learnable weights.
    """
    
    def __init__(self, models, learnable_weights=True):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if learnable_weights:
            # Learnable ensemble weights
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        else:
            # Equal weights
            self.register_buffer('weights', torch.ones(self.num_models) / self.num_models)
    
    def forward(self, x):
        """
        Forward pass through all models and combine predictions.
        
        Args:
            x: Input tensor
        
        Returns:
            Ensemble logits
        """
        # Get predictions from all models
        logits_list = []
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)
        
        # Stack: (num_models, batch, num_classes)
        logits_stack = torch.stack(logits_list)
        
        # Softmax normalize weights
        weights = F.softmax(self.weights, dim=0)
        
        # Weighted average
        # (num_models, 1, 1) * (num_models, batch, num_classes)
        ensemble_logits = torch.sum(
            weights.view(-1, 1, 1) * logits_stack,
            dim=0
        )
        
        return ensemble_logits
    
    def predict_with_confidence(self, x):
        """
        Predict with confidence estimation based on model agreement.
        
        Returns:
            predictions, confidence_scores
        """
        self.eval()
        with torch.no_grad():
            logits_list = []
            for model in self.models:
                logits = model(x)
                logits_list.append(logits)
            
            logits_stack = torch.stack(logits_list)  # (num_models, batch, num_classes)
            probs_stack = F.softmax(logits_stack, dim=-1)
            
            # Model agreement (lower variance = higher confidence)
            mean_probs = probs_stack.mean(dim=0)
            var_probs = probs_stack.var(dim=0)
            confidence = 1 - var_probs.mean(dim=-1)  # Higher confidence = lower variance
            
            predictions = mean_probs.argmax(dim=-1)
            
        return predictions, confidence


def train_diverse_models(
    data_dirs,
    num_models=5,
    epochs=20,
    batch_size=128,
    device='cuda'
):
    """
    Train diverse ensemble of models.
    
    Diversity through:
    1. Different architectures
    2. Different chart types
    3. Different random seeds
    """
    print("="*70)
    print(f"Training Ensemble of {num_models} Diverse Models")
    print("="*70)
    
    # Model configurations for diversity
    configs = [
        {'arch': 'resnet18', 'chart_type': 'ohlc', 'seed': 42},
        {'arch': 'resnet18', 'chart_type': 'gaf', 'seed': 123},
        {'arch': 'efficientnet_b0', 'chart_type': 'ohlc', 'seed': 456},
        {'arch': 'resnet18', 'chart_type': 'hybrid', 'seed': 789},
        {'arch': 'resnet18', 'chart_type': 'ohlc', 'seed': 101, 'augment': 'heavy'},
    ][:num_models]
    
    models = []
    histories = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"Training Model {i+1}/{num_models}")
        print(f"Config: {config}")
        print(f"{'='*70}")
        
        # Set seed
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # Create dataset
        train_ds = StockDataset(
            data_dir=data_dirs,
            window_size=10,
            prediction_horizon=5,
            img_size=(256, 256),
            mode='train',
            chart_type=config['chart_type'],
            augment_prob=0.4 if config.get('augment') == 'heavy' else 0.3,
        )
        
        val_ds = StockDataset(
            data_dir=data_dirs,
            window_size=10,
            prediction_horizon=5,
            img_size=(256, 256),
            mode='val',
            chart_type=config['chart_type'],
            augment_prob=0.0,
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        # Create model
        input_channels = train_ds.get_num_channels()
        model = CNNModel(
            num_classes=2,
            input_channels=input_channels,
            arch=config['arch'],
            pretrained=True
        ).to(device)
        
        # Train
        criterion = nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0
        history = []
        
        for epoch in range(epochs):
            model.train()
            train_loss, train_acc = 0, 0
            
            for batch in train_loader:
                imgs, seqs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += (outputs.argmax(1) == labels).float().mean().item()
            
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    imgs, seqs, labels = batch
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (outputs.argmax(1) == labels).float().mean().item()
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            
            scheduler.step()
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            })
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: val_acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        print(f"  Best val acc: {best_val_acc:.4f}")
        
        models.append(model)
        histories.append(history)
    
    return models, histories


def train_multiscale_ensemble(
    data_dirs,
    epochs=25,
    batch_size=64,
    device='cuda'
):
    """
    Train multi-scale ensemble models.
    """
    print("="*70)
    print("Training Multi-Scale Ensemble")
    print("="*70)
    
    from src.data.multiscale_dataset import MultiScaleStockDataset, collate_multiscale
    
    # Create multi-scale dataset
    train_ds = MultiScaleStockDataset(
        data_dir=data_dirs,
        scales=[5, 10, 20],
        prediction_horizon=5,
        img_size=(256, 256),
        mode='train',
        chart_type='ohlc',
        augment_prob=0.3,
    )
    
    val_ds = MultiScaleStockDataset(
        data_dir=data_dirs,
        scales=[5, 10, 20],
        prediction_horizon=5,
        img_size=(256, 256),
        mode='val',
        chart_type='ohlc',
        augment_prob=0.0,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_multiscale
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_multiscale
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Train multiple multi-scale models with different strategies
    models = []
    
    # Model 1: Multi-scale with attention
    print("\nTraining Multi-Scale Attention Model...")
    model1 = MultiScaleKLineEncoder(
        num_classes=2,
        input_channels=3,
        embedding_dim=256,
        pretrained=True
    ).to(device)
    train_single_multiscale(model1, train_loader, val_loader, epochs, device)
    models.append(('attention', model1))
    
    # Model 2: Hierarchical fusion
    print("\nTraining Hierarchical Multi-Scale Model...")
    model2 = HierarchicalMultiScaleCNN(
        num_classes=2,
        input_channels=3,
        pretrained=True
    ).to(device)
    train_single_multiscale(model2, train_loader, val_loader, epochs, device)
    models.append(('hierarchical', model2))
    
    # Model 3: Lightweight with ensemble heads
    print("\nTraining Lightweight Multi-Scale Model...")
    model3 = LightweightMultiScaleCNN(
        num_classes=2,
        input_channels=3,
        pretrained=True
    ).to(device)
    train_single_multiscale(model3, train_loader, val_loader, epochs, device)
    models.append(('lightweight', model3))
    
    return models


def train_single_multiscale(model, train_loader, val_loader, epochs, device):
    """Train a single multi-scale model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0
        
        for images, labels in train_loader:
            # images is dict with '5d', '10d', '20d'
            labels = labels.to(device)
            images = {k: v.to(device) for k, v in images.items()}
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(device)
                images = {k: v.to(device) for k, v in images.items()}
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: val_acc={val_acc:.4f}")
    
    print(f"  Best val acc: {best_val_acc:.4f}")


def evaluate_ensemble(models, test_loader, device, ensemble_type='average'):
    """
    Evaluate ensemble on test set.
    
    Args:
        models: List of (name, model) tuples or just models
        test_loader: Test data loader
        device: Device
        ensemble_type: 'average', 'weighted', or 'voting'
    """
    print("\n" + "="*70)
    print("Ensemble Evaluation")
    print("="*70)
    
    for model in models:
        if isinstance(model, tuple):
            name, model = model
        else:
            name = 'model'
        model.eval()
    
    all_preds = defaultdict(list)
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch[0], dict):
                # Multi-scale
                images, labels = batch
                images = {k: v.to(device) for k, v in images.items()}
            else:
                images, seqs, labels = batch
                images = images.to(device)
            
            labels = labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            
            # Get predictions from each model
            for i, model in enumerate(models):
                if isinstance(model, tuple):
                    name, model = model
                else:
                    name = f'model_{i}'
                
                outputs = model(images)
                probs = F.softmax(outputs, dim=-1)
                all_preds[name].extend(probs.cpu().numpy())
    
    # Convert to arrays
    all_labels = np.array(all_labels)
    individual_accs = {}
    
    print("\nIndividual Model Performance:")
    for name, probs in all_preds.items():
        preds = np.array(probs).argmax(axis=1)
        acc = (preds == all_labels).mean()
        individual_accs[name] = acc
        print(f"  {name}: {acc:.4f}")
    
    # Ensemble methods
    print("\nEnsemble Performance:")
    
    # Average ensemble
    avg_probs = np.mean(list(all_preds.values()), axis=0)
    avg_preds = avg_probs.argmax(axis=1)
    avg_acc = (avg_preds == all_labels).mean()
    print(f"  Average Ensemble: {avg_acc:.4f}")
    
    # Weighted ensemble (by validation accuracy - placeholder)
    weights = np.array([acc for acc in individual_accs.values()])
    weights = weights / weights.sum()
    weighted_probs = np.average(list(all_preds.values()), axis=0, weights=weights)
    weighted_preds = weighted_probs.argmax(axis=1)
    weighted_acc = (weighted_preds == all_labels).mean()
    print(f"  Weighted Ensemble: {weighted_acc:.4f}")
    
    # Majority voting
    all_predictions = np.array([np.array(probs).argmax(axis=1) for probs in all_preds.values()])
    vote_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)
    vote_acc = (vote_preds == all_labels).mean()
    print(f"  Voting Ensemble: {vote_acc:.4f}")
    
    return {
        'individual': individual_accs,
        'average': avg_acc,
        'weighted': weighted_acc,
        'voting': vote_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Ensemble Models')
    parser.add_argument('--type', choices=['diverse', 'multiscale', 'both'], default='both')
    parser.add_argument('--markets', nargs='+', default=['us'])
    parser.add_argument('--num-models', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare data directories
    data_dirs = [str(PROJECT_ROOT / 'data' / 'raw' / m) for m in args.markets]
    
    all_models = []
    
    if args.type in ['diverse', 'both']:
        # Train diverse ensemble
        models, histories = train_diverse_models(
            data_dirs=data_dirs,
            num_models=args.num_models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        all_models.extend([(f'diverse_{i}', m) for i, m in enumerate(models)])
    
    if args.type in ['multiscale', 'both']:
        # Train multi-scale ensemble
        models = train_multiscale_ensemble(
            data_dirs=data_dirs,
            epochs=args.epochs,
            batch_size=args.batch_size // 2,
            device=device
        )
        all_models.extend(models)
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final Ensemble Evaluation")
    print("="*70)
    
    # Create test loader
    test_ds = StockDataset(
        data_dir=data_dirs,
        window_size=10,
        prediction_horizon=5,
        img_size=(256, 256),
        mode='test',
        chart_type='ohlc',
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Filter to single-scale models for evaluation
    single_scale_models = [(name, m) for name, m in all_models 
                           if not isinstance(m, (MultiScaleKLineEncoder, 
                                                HierarchicalMultiScaleCNN,
                                                LightweightMultiScaleCNN))]
    
    if single_scale_models:
        results = evaluate_ensemble(
            [m for _, m in single_scale_models],
            test_loader,
            device
        )
        
        # Save results
        output_dir = settings.MODELS_DIR / 'ensemble'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_dir / 'ensemble_results.json'}")
    
    print("\nEnsemble training complete!")


if __name__ == '__main__':
    main()
