"""
Grouped Training with Sparse CNN + Quantile Filtering

Key improvements over v2 baseline:
1. Quantile-based flat-sample filtering (remove middle 35% ambiguous returns)
2. Sparse grayscale OHLC images (64x60 for 20-day window)
3. Lightweight 3-layer CNN (~709K params vs 11M ResNet18)
4. Preserves sector-based grouped training

Usage:
    python scripts/run_sparse_grouped_training.py
    python scripts/run_sparse_grouped_training.py --window-size 20 --quantile-filter 0.35
    python scripts/run_sparse_grouped_training.py --groups Tech_Semiconductors Tech_Software
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.sparse_cnn import SparseCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'grouped_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPARSE_CONFIG = {
    'arch': 'sparse_cnn',
    'window_size': 20,
    'chart_type': 'sparse',
    'grayscale': True,
    'quantile_filter': 0.35,       # Filter middle 35% of ambiguous samples
    'label_threshold': 'none',     # Use quantile filter instead
    'use_clahe': False,
    'output_channels': 'rgb',      # Ignored for sparse chart_type (always 1-ch)
    'augment_prob': 0.0,
    'mixup_prob': 0.0,
    'norm_method': 'minmax',       # Ignored for sparse chart_type
    'img_size': (64, 60),          # Ignored for sparse chart_type (fixed per window)
    'epochs': 30,
    'batch_size': 128,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'patience': 8,
    'num_runs': 3,
}


def load_stock_groups():
    group_file = PROJECT_ROOT / 'data' / 'us_stock_groups.json'
    with open(group_file, 'r') as f:
        data = json.load(f)
    return data['groups']


def create_global_datasets(config):
    """Create train/val/test datasets once for all groups."""
    common = {
        'data_dir': str(PROJECT_ROOT / 'data' / 'raw' / 'us'),
        'window_size': config['window_size'],
        'prediction_horizon': 5,
        'img_size': config['img_size'],
        'norm_method': config['norm_method'],
        'use_clahe': config['use_clahe'],
        'output_channels': config['output_channels'],
        'label_threshold': config['label_threshold'],
        'chart_type': config['chart_type'],
        'grayscale': config['grayscale'],
    }

    print("Loading global datasets...", flush=True)

    # Train set gets quantile filtering
    train_ds = StockDataset(
        mode='train',
        augment_prob=config['augment_prob'],
        mixup_prob=config['mixup_prob'],
        quantile_filter=config['quantile_filter'],
        **common,
    )
    # Val/test: no filtering (unbiased evaluation)
    val_ds = StockDataset(mode='val', augment_prob=0.0, **common)
    test_ds = StockDataset(mode='test', augment_prob=0.0, **common)

    print(f"Global datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)
    print(f"Stocks loaded: {sorted(train_ds.data_cache.keys())}", flush=True)

    # Print quantile filter summary if active
    if train_ds.quantile_calculator is not None:
        train_ds.quantile_calculator.summary()

    return train_ds, val_ds, test_ds


def filter_dataset_by_stocks(dataset, stock_list):
    stock_set = set(stock_list)
    return [i for i, s in enumerate(dataset.samples) if s['ticker'] in stock_set]


def create_group_dataloaders(train_ds, val_ds, test_ds, group_stocks, config):
    train_idx = filter_dataset_by_stocks(train_ds, group_stocks)
    val_idx = filter_dataset_by_stocks(val_ds, group_stocks)
    test_idx = filter_dataset_by_stocks(test_ds, group_stocks)

    train_sub = Subset(train_ds, train_idx)
    val_sub = Subset(val_ds, val_idx)
    test_sub = Subset(test_ds, test_idx)

    kw = {'batch_size': config['batch_size'], 'num_workers': 4,
          'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True}

    return (train_sub, val_sub, test_sub,
            DataLoader(train_sub, shuffle=True, drop_last=True, **kw),
            DataLoader(val_sub, shuffle=False, **kw),
            DataLoader(test_sub, shuffle=False, **kw))


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []

    for imgs, _, lbls in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(imgs)
        loss = criterion(out, lbls)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds_all.extend(out.argmax(1).detach().cpu().numpy())
        labels_all.extend(lbls.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(labels_all, preds_all) if labels_all else 0
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all, probs_all = 0, [], [], []

    with torch.no_grad():
        for imgs, _, lbls in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)

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


def run_single_group(group_name, group_stocks, train_ds, val_ds, test_ds, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  [{group_name}][seed={seed}] Creating dataloaders...", flush=True)
    train_sub, val_sub, test_sub, train_loader, val_loader, test_loader = \
        create_group_dataloaders(train_ds, val_ds, test_ds, group_stocks, config)

    n_train, n_val, n_test = len(train_sub), len(val_sub), len(test_sub)
    print(f"  [{group_name}][seed={seed}] Data: train={n_train}, val={n_val}, test={n_test}", flush=True)

    if n_train == 0 or n_val == 0 or n_test == 0:
        print(f"  [{group_name}][seed={seed}] Skipping — insufficient data", flush=True)
        return None

    # Create lightweight Sparse CNN
    model = SparseCNN(
        num_classes=2,
        window_size=config['window_size'],
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{group_name}][seed={seed}] Model: SparseCNN, params={param_count:,}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_f1, best_state, patience_cnt, history = 0, None, 0, []

    print(f"  [{group_name}][seed={seed}] Training...", flush=True)
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_m['accuracy'],
            'val_f1': val_m['f1'],
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{group_name}][seed={seed}] Epoch {epoch+1}/{config['epochs']}: "
                  f"loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_f1={val_m['f1']:.4f}, val_acc={val_m['accuracy']:.4f}", flush=True)

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                print(f"  [{group_name}][seed={seed}] Early stop at epoch {epoch+1}", flush=True)
                break

    # Load best model and evaluate on test set
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_m = evaluate(model, test_loader, criterion)

    print(f"  [{group_name}][seed={seed}] Test: acc={test_m['accuracy']:.4f}, "
          f"f1={test_m['f1']:.4f}, auc={test_m['auc']:.4f}", flush=True)

    return {
        'seed': seed,
        'test_metrics': test_m,
        'best_val_f1': best_f1,
        'epochs_trained': len(history),
        'history': history,
        'num_stocks': len(group_stocks),
        'train_samples': n_train,
        'val_samples': n_val,
        'test_samples': n_test,
        'param_count': param_count,
    }


def main():
    parser = argparse.ArgumentParser(description='Sparse CNN grouped training')
    parser.add_argument('--window-size', type=int, default=20, choices=[5, 20, 60])
    parser.add_argument('--quantile-filter', type=float, default=0.35)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-runs', type=int, default=3)
    parser.add_argument('--groups', nargs='+', default=None,
                        help='Train only these groups (default: all)')
    args = parser.parse_args()

    config = deepcopy(SPARSE_CONFIG)
    config['window_size'] = args.window_size
    config['quantile_filter'] = args.quantile_filter
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['num_runs'] = args.num_runs

    # Update image size for the chosen window
    from src.models.sparse_cnn import SPARSE_IMAGE_SIZES
    config['img_size'] = SPARSE_IMAGE_SIZES.get(args.window_size, (64, 60))

    print("=" * 80, flush=True)
    print("Sparse CNN Grouped Training (Quantile Filter + Lightweight CNN)", flush=True)
    print("=" * 80, flush=True)
    print(f"Start: {datetime.now()}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Window: {config['window_size']} days", flush=True)
    print(f"Image size: {config['img_size']}", flush=True)
    print(f"Quantile filter: {config['quantile_filter']}", flush=True)
    print(f"Epochs: {config['epochs']}, Batch: {config['batch_size']}, LR: {config['lr']}", flush=True)
    print(f"Runs per group: {config['num_runs']}", flush=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load stock groups
    stock_groups = load_stock_groups()
    if args.groups:
        stock_groups = {k: v for k, v in stock_groups.items() if k in args.groups}

    print(f"\n{len(stock_groups)} industry groups:", flush=True)
    for gid, ginfo in stock_groups.items():
        print(f"  - {gid}: {ginfo['name']} ({len(ginfo['stocks'])} stocks)", flush=True)

    # Create global datasets once
    train_ds, val_ds, test_ds = create_global_datasets(config)

    # Train each group
    all_results = {}
    seeds = [42, 142, 242, 342, 442][:config['num_runs']]

    for group_id, group_info in stock_groups.items():
        print(f"\n{'='*80}", flush=True)
        print(f"Group: {group_id} ({group_info['name']})", flush=True)
        print(f"{'='*80}", flush=True)

        runs = []
        for i, seed in enumerate(seeds):
            print(f"\n--- Run {i+1}/{config['num_runs']} ---", flush=True)
            result = run_single_group(
                group_id, group_info['stocks'],
                train_ds, val_ds, test_ds,
                config, seed,
            )
            if result is not None:
                runs.append(result)

        if not runs:
            print(f"  [{group_id}] No valid runs", flush=True)
            continue

        # Summary stats
        summary = {}
        for k in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
            vals = [r['test_metrics'][k] for r in runs]
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))

        print(f"\n=== {group_id} Summary ===", flush=True)
        print(f"  F1:       {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}", flush=True)
        print(f"  Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}", flush=True)
        print(f"  AUC:      {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}", flush=True)

        all_results[group_id] = {
            'group_name': group_info['name'],
            'description': group_info['description'],
            'num_stocks': len(group_info['stocks']),
            'stocks': group_info['stocks'],
            'runs': runs,
            'summary': summary,
        }

    # Overall summary
    print(f"\n{'='*80}", flush=True)
    print("Overall Summary (All Groups)", flush=True)
    print(f"{'='*80}", flush=True)

    for group_id, result in all_results.items():
        s = result['summary']
        print(f"  {group_id:25s}: F1={s['f1_mean']:.4f}+/-{s['f1_std']:.4f}, "
              f"Acc={s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.4f}", flush=True)

    # Weighted average F1
    all_f1s, all_weights = [], []
    for result in all_results.values():
        for run in result['runs']:
            all_f1s.append(run['test_metrics']['f1'])
            all_weights.append(run['test_samples'])

    if all_f1s:
        weighted_avg_f1 = float(np.average(all_f1s, weights=all_weights))
        simple_avg_f1 = float(np.mean(all_f1s))
        baseline_f1 = 0.5083

        print(f"\nWeighted Average F1: {weighted_avg_f1:.4f}", flush=True)
        print(f"Simple Average F1:  {simple_avg_f1:.4f}", flush=True)
        print(f"Baseline (v2):      {baseline_f1:.4f}", flush=True)
        print(f"Improvement:        {(weighted_avg_f1 - baseline_f1) * 100:+.2f}%", flush=True)
    else:
        weighted_avg_f1 = 0
        simple_avg_f1 = 0

    # Save results
    output = {
        'experiment': 'Sparse CNN Grouped Training (Quantile Filter + Lightweight CNN)',
        'config': config,
        'baseline_f1': 0.5083,
        'weighted_avg_f1': weighted_avg_f1,
        'simple_avg_f1': simple_avg_f1,
        'num_groups': len(all_results),
        'groups': all_results,
        'timestamp': datetime.now().isoformat(),
    }

    output_file = OUTPUT_DIR / f"sparse_grouped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}", flush=True)
    print(f"End: {datetime.now()}", flush=True)


if __name__ == '__main__':
    main()
