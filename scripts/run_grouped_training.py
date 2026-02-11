"""Grouped Training Experiment - Train separate models for each industry group"""
import os, sys, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'grouped_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use same config as KLineNet-MC (best baseline)
BASE_CONFIG = {
    'model_type': 'cnn',
    'img_size': (128, 128),
    'norm_method': 'robust',
    'use_clahe': True,
    'output_channels': 'rgb+edge',
    'label_threshold': 'dynamic',
    'augment_prob': 0.3,
    'mixup_prob': 0.1,
    'epochs': 20,
    'batch_size': 64,  # Reduced from 128 to save GPU memory
    'lr': 4e-4,
    'weight_decay': 1e-5,
    'patience': 5,
    'num_runs': 3,
}

def load_stock_groups():
    """Load stock group definitions from JSON file"""
    group_file = PROJECT_ROOT / 'data' / 'us_stock_groups.json'
    with open(group_file, 'r') as f:
        data = json.load(f)
    return data['groups']

def filter_dataset_by_stocks(dataset, stock_list):
    """
    Filter dataset to only include samples from specified stocks.
    Returns filtered indices that can be used with Subset.
    """
    stock_set = set(stock_list)
    filtered_indices = [i for i, sample in enumerate(dataset.samples)
                       if sample['ticker'] in stock_set]
    return filtered_indices

def create_group_dataloaders(group_stocks, config):
    """Create dataloaders for a specific stock group"""
    common = {
        'data_dir': str(PROJECT_ROOT / 'data' / 'raw' / 'us'),  # US stocks only
        'window_size': 20, 'prediction_horizon': 5,
        'img_size': config['img_size'], 'norm_method': config['norm_method'],
        'use_clahe': config['use_clahe'], 'output_channels': config['output_channels'],
        'label_threshold': config['label_threshold'],
    }

    # Load full datasets first
    train_ds_full = StockDataset(mode='train', augment_prob=config['augment_prob'],
                                 mixup_prob=config['mixup_prob'], **common)
    val_ds_full = StockDataset(mode='val', augment_prob=0.0, **common)
    test_ds_full = StockDataset(mode='test', augment_prob=0.0, **common)

    # Filter by group stocks
    train_indices = filter_dataset_by_stocks(train_ds_full, group_stocks)
    val_indices = filter_dataset_by_stocks(val_ds_full, group_stocks)
    test_indices = filter_dataset_by_stocks(test_ds_full, group_stocks)

    # Create subset datasets
    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)
    test_ds = Subset(test_ds_full, test_indices)

    # Create dataloaders
    loader_kw = {'batch_size': config['batch_size'], 'num_workers': 4,
                 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True}

    return (train_ds_full, val_ds_full, test_ds_full,
            train_ds, val_ds, test_ds,
            DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw),
            DataLoader(val_ds, shuffle=False, **loader_kw),
            DataLoader(test_ds, shuffle=False, **loader_kw))

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, preds, labels = 0, [], []
    for imgs, _, lbls in loader:
        imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=True):
            out = model(imgs)
            loss = criterion(out, lbls)
        if torch.isnan(loss): continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds.extend(out.argmax(1).detach().cpu().numpy())
        labels.extend(lbls.cpu().numpy())
    return total_loss / max(len(loader), 1), accuracy_score(labels, preds) if labels else 0

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, labels, probs = 0, [], [], []
    with torch.no_grad():
        for imgs, _, lbls in loader:
            imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
            with autocast(device_type='cuda', enabled=True):
                out = model(imgs)
                loss = criterion(out, lbls)
            if not torch.isnan(loss): total_loss += loss.item()
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.cpu().numpy())
            probs.extend(torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy())
    metrics = {'loss': total_loss / max(len(loader), 1), 'accuracy': accuracy_score(labels, preds),
               'f1': f1_score(labels, preds, average='weighted'),
               'precision': precision_score(labels, preds, average='weighted', zero_division=0),
               'recall': recall_score(labels, preds, average='weighted', zero_division=0)}
    try: metrics['auc'] = roc_auc_score(labels, probs)
    except: metrics['auc'] = 0.5
    return metrics

def run_single_group(group_name, group_stocks, config, seed):
    """Train a single model for one stock group"""
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"  [{group_name}][seed={seed}] Loading data...", flush=True)

    train_ds_full, val_ds_full, test_ds_full, train_ds, val_ds, test_ds, \
        train_loader, val_loader, test_loader = create_group_dataloaders(group_stocks, config)

    print(f"  [{group_name}][seed={seed}] Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)
    print(f"  [{group_name}][seed={seed}] Stocks in group: {len(group_stocks)}", flush=True)

    # Create model with same architecture as baseline
    model = CNNModel(num_classes=2, input_channels=train_ds_full.get_num_channels(),
                     arch='resnet18', pretrained=False).to(DEVICE)
    # Disable torch.compile to save memory
    # if hasattr(torch, 'compile'):
    #     try: model = torch.compile(model, mode='reduce-overhead')
    #     except: pass

    criterion = nn.CrossEntropyLoss(weight=train_ds_full.get_class_weights().to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler('cuda')

    best_f1, best_state, patience_cnt, history = 0, None, 0, []
    print(f"  [{group_name}][seed={seed}] Training...", flush=True)
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step()
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
                       'val_acc': val_m['accuracy'], 'val_f1': val_m['f1']})
        if (epoch + 1) % 5 == 0:
            print(f"  [{group_name}][seed={seed}] Epoch {epoch+1}/{config['epochs']}: "
                  f"loss={train_loss:.4f}, acc={train_acc:.4f}, val_f1={val_m['f1']:.4f}", flush=True)
        if val_m['f1'] > best_f1:
            best_f1, best_state, patience_cnt = val_m['f1'], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                print(f"  [{group_name}][seed={seed}] Early stop at epoch {epoch+1}", flush=True)
                break

    if best_state: model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_m = evaluate(model, test_loader, criterion)
    print(f"  [{group_name}][seed={seed}] Test: acc={test_m['accuracy']:.4f}, f1={test_m['f1']:.4f}, auc={test_m['auc']:.4f}", flush=True)

    return {
        'seed': seed,
        'test_metrics': test_m,
        'best_val_f1': best_f1,
        'epochs_trained': len(history),
        'history': history,
        'num_stocks': len(group_stocks),
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': len(test_ds)
    }

def main():
    print("="*80 + f"\nGrouped Training Experiment (US Stocks by Industry)\n" + "="*80, flush=True)
    print(f"Start: {datetime.now()}, Device: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load stock groups
    stock_groups = load_stock_groups()
    print(f"\nLoaded {len(stock_groups)} industry groups:", flush=True)
    for gid, ginfo in stock_groups.items():
        print(f"  - {gid}: {ginfo['name']} ({len(ginfo['stocks'])} stocks)", flush=True)

    # Train each group
    all_results = {}
    for group_id, group_info in stock_groups.items():
        print(f"\n{'='*80}\nTraining Group: {group_id} ({group_info['name']})\n{'='*80}", flush=True)

        group_config = deepcopy(BASE_CONFIG)
        group_config['name'] = f"Grouped_{group_id}"

        runs = []
        for i in range(group_config['num_runs']):
            print(f"\n--- Run {i+1}/{group_config['num_runs']} ---", flush=True)
            runs.append(run_single_group(group_id, group_info['stocks'], group_config, 42 + i * 100))

        # Compute summary stats
        summary = {}
        for k in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
            vals = [r['test_metrics'][k] for r in runs]
            summary[f'{k}_mean'], summary[f'{k}_std'] = float(np.mean(vals)), float(np.std(vals))

        print(f"\n=== {group_id} Summary ===", flush=True)
        print(f"F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}", flush=True)
        print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}", flush=True)
        print(f"AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}", flush=True)

        all_results[group_id] = {
            'group_name': group_info['name'],
            'description': group_info['description'],
            'num_stocks': len(group_info['stocks']),
            'stocks': group_info['stocks'],
            'config': group_config,
            'runs': runs,
            'summary': summary
        }

    # Overall summary across all groups
    print(f"\n{'='*80}\nOverall Summary (All Groups)\n{'='*80}", flush=True)
    for group_id, result in all_results.items():
        s = result['summary']
        print(f"{group_id:25s}: F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}, "
              f"Acc={s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}, "
              f"Stocks={result['num_stocks']}", flush=True)

    # Compute weighted average (by number of test samples)
    all_f1s, all_weights = [], []
    for group_id, result in all_results.items():
        for run in result['runs']:
            all_f1s.append(run['test_metrics']['f1'])
            all_weights.append(run['test_samples'])

    weighted_avg_f1 = np.average(all_f1s, weights=all_weights)
    simple_avg_f1 = np.mean(all_f1s)

    print(f"\nWeighted Average F1 (by test samples): {weighted_avg_f1:.4f}", flush=True)
    print(f"Simple Average F1: {simple_avg_f1:.4f}", flush=True)
    print(f"Baseline (KLineNet-MC mixed training): 0.5083", flush=True)
    print(f"Improvement: {(weighted_avg_f1 - 0.5083) * 100:+.2f}%", flush=True)

    # Save results
    output = {
        'experiment': 'Grouped Training by Industry',
        'baseline_f1': 0.5083,
        'weighted_avg_f1': float(weighted_avg_f1),
        'simple_avg_f1': float(simple_avg_f1),
        'improvement_pct': float((weighted_avg_f1 - 0.5083) * 100),
        'num_groups': len(stock_groups),
        'groups': all_results,
        'timestamp': datetime.now().isoformat()
    }

    output_file = OUTPUT_DIR / f"grouped_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}", flush=True)
    print(f"End: {datetime.now()}", flush=True)

if __name__ == '__main__':
    main()
