"""
Parallel Experiment Runner for ECCV Paper
Runs multiple models concurrently to save time.
"""
import os
import sys
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel
from models.baselines import LSTMEncoder, ResNet1DEncoder

# Configs for each model
EXPERIMENTS = {
    'LSTM': {
        'model_type': 'lstm',
        'use_gpu': False,  # Run on CPU to avoid NaN issues
    },
    'ResNet1D': {
        'model_type': 'resnet1d',
        'use_gpu': False,  # Run on CPU
    },
    'CNN-Raw': {
        'model_type': 'cnn',
        'img_size': (64, 64),
        'norm_method': 'minmax',
        'use_clahe': False,
        'output_channels': 'rgb',
        'label_threshold': 0.005,
        'augment_prob': 0.0,
        'use_gpu': True,
    },
    'KLineNet': {
        'model_type': 'cnn',
        'img_size': (128, 128),
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'use_gpu': True,
    },
    'KLineNet-MC': {
        'model_type': 'cnn',
        'img_size': (128, 128),
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb+edge',
        'label_threshold': 'dynamic',
        'augment_prob': 0.3,
        'mixup_prob': 0.1,
        'use_gpu': True,
    },
}

DATA_DIRS = [str(PROJECT_ROOT / 'data' / 'raw' / 'us'), str(PROJECT_ROOT / 'data' / 'raw' / 'cn')]
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eccv_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(name, config, seed=42):
    """Run a single experiment."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{name}] Starting experiment (seed={seed})", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if config.get('use_gpu', True) and torch.cuda.is_available() else 'cpu')
    print(f"[{name}] Device: {device}", flush=True)

    # Create datasets
    print(f"[{name}] Loading data...", flush=True)
    common_args = {
        'data_dir': DATA_DIRS,
        'window_size': 20,
        'prediction_horizon': 5,
    }

    if config['model_type'] == 'cnn':
        common_args.update({
            'img_size': config.get('img_size', (128, 128)),
            'norm_method': config.get('norm_method', 'robust'),
            'use_clahe': config.get('use_clahe', True),
            'output_channels': config.get('output_channels', 'rgb'),
            'label_threshold': config.get('label_threshold', 'dynamic'),
        })

    train_ds = StockDataset(mode='train', augment_prob=config.get('augment_prob', 0.0),
                            mixup_prob=config.get('mixup_prob', 0.0), **common_args)
    val_ds = StockDataset(mode='val', augment_prob=0.0, **common_args)
    test_ds = StockDataset(mode='test', augment_prob=0.0, **common_args)

    print(f"[{name}] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Create model
    if config['model_type'] == 'cnn':
        input_channels = train_ds.get_num_channels()
        model = CNNModel(num_classes=2, input_channels=input_channels,
                        arch='resnet18', pretrained=False).to(device)
    elif config['model_type'] == 'lstm':
        model = LSTMEncoder(input_dim=2, hidden_dim=128, num_layers=2,
                           embedding_dim=256, num_classes=2).to(device)
    elif config['model_type'] == 'resnet1d':
        model = ResNet1DEncoder(input_dim=2, layers=[2,2,2,2],
                               embedding_dim=256, num_classes=2).to(device)

    # Training setup
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Training loop
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    patience = 5
    history = []

    print(f"[{name}] Training...", flush=True)
    for epoch in range(20):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            if config['model_type'] == 'cnn':
                imgs, seqs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
            else:
                imgs, seqs, labels = batch
                seqs, labels = seqs.to(device), labels.to(device)
                optimizer.zero_grad()
                if config['model_type'] == 'lstm':
                    _, outputs = model(seqs)
                else:
                    seqs = seqs.permute(0, 2, 1)
                    _, outputs = model(seqs)

            loss = criterion(outputs, labels)

            # Handle NaN loss
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                if config['model_type'] == 'cnn':
                    imgs, seqs, labels = batch
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                else:
                    imgs, seqs, labels = batch
                    seqs, labels = seqs.to(device), labels.to(device)
                    if config['model_type'] == 'lstm':
                        _, outputs = model(seqs)
                    else:
                        seqs = seqs.permute(0, 2, 1)
                        _, outputs = model(seqs)

                probs = torch.softmax(outputs, dim=1)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"[{name}] Epoch {epoch+1}/20: loss={avg_loss:.4f}, acc={train_acc:.4f}, "
              f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}", flush=True)

        history.append({'epoch': epoch+1, 'loss': avg_loss, 'train_acc': train_acc,
                       'val_acc': val_acc, 'val_f1': val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{name}] Early stopping at epoch {epoch+1}", flush=True)
                break

    # Test evaluation
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if config['model_type'] == 'cnn':
                imgs, seqs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
            else:
                imgs, seqs, labels = batch
                seqs, labels = seqs.to(device), labels.to(device)
                if config['model_type'] == 'lstm':
                    _, outputs = model(seqs)
                else:
                    seqs = seqs.permute(0, 2, 1)
                    _, outputs = model(seqs)

            probs = torch.softmax(outputs, dim=1)
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except:
        test_auc = 0.5

    result = {
        'name': name,
        'seed': seed,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'best_val_f1': best_val_f1,
        'epochs_trained': len(history),
        'history': history,
    }

    print(f"\n[{name}] === RESULTS ===", flush=True)
    print(f"[{name}] Test Accuracy: {test_acc:.4f}", flush=True)
    print(f"[{name}] Test F1 Score: {test_f1:.4f}", flush=True)
    print(f"[{name}] Test AUC: {test_auc:.4f}", flush=True)

    return result


def run_all_experiments(num_runs=1):
    """Run all experiments."""
    print("="*80, flush=True)
    print("ECCV Experiments - Parallel Runner", flush=True)
    print(f"Models: {list(EXPERIMENTS.keys())}", flush=True)
    print(f"Runs per model: {num_runs}", flush=True)
    print("="*80, flush=True)

    all_results = {}

    for name, config in EXPERIMENTS.items():
        run_results = []
        for run_id in range(num_runs):
            seed = 42 + run_id * 100
            result = run_experiment(name, config, seed=seed)
            run_results.append(result)

        # Aggregate
        accs = [r['test_accuracy'] for r in run_results]
        f1s = [r['test_f1'] for r in run_results]
        aucs = [r['test_auc'] for r in run_results]

        all_results[name] = {
            'runs': run_results,
            'summary': {
                'accuracy_mean': np.mean(accs),
                'accuracy_std': np.std(accs),
                'f1_mean': np.mean(f1s),
                'f1_std': np.std(f1s),
                'auc_mean': np.mean(aucs),
                'auc_std': np.std(aucs),
            }
        }

    # Print final table
    print("\n" + "="*80, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*80, flush=True)
    print(f"{'Model':<20} {'Accuracy':<18} {'F1 Score':<18} {'AUC':<18}", flush=True)
    print("-"*80, flush=True)

    for name, data in all_results.items():
        s = data['summary']
        print(f"{name:<20} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}    "
              f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}    "
              f"{s['auc_mean']:.4f}±{s['auc_std']:.4f}", flush=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"parallel_results_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to: {output_file}", flush=True)
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per model')
    args = parser.parse_args()

    run_all_experiments(num_runs=args.runs)
