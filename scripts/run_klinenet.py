"""KLineNet Experiment - Our method with CLAHE + robust norm + augmentation"""
import os, sys, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eccv_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'name': 'KLineNet',
    'model_type': 'cnn',
    'img_size': (128, 128),
    'norm_method': 'robust',
    'use_clahe': True,
    'output_channels': 'rgb',
    'label_threshold': 'dynamic',
    'augment_prob': 0.3,
    'epochs': 20,
    'batch_size': 128,
    'lr': 4e-4,
    'weight_decay': 1e-5,
    'patience': 5,
    'num_runs': 3,
}

def create_dataloaders():
    common = {
        'data_dir': [str(PROJECT_ROOT / 'data' / 'raw' / 'us')],  # Only US stocks
        'window_size': 20, 'prediction_horizon': 5,
        'img_size': CONFIG['img_size'], 'norm_method': CONFIG['norm_method'],
        'use_clahe': CONFIG['use_clahe'], 'output_channels': CONFIG['output_channels'],
        'label_threshold': CONFIG['label_threshold'],
    }
    train_ds = StockDataset(mode='train', augment_prob=CONFIG['augment_prob'], **common)
    val_ds = StockDataset(mode='val', augment_prob=0.0, **common)
    test_ds = StockDataset(mode='test', augment_prob=0.0, **common)
    
    loader_kw = {'batch_size': CONFIG['batch_size'], 'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True}
    return train_ds, val_ds, test_ds, \
           DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw), \
           DataLoader(val_ds, shuffle=False, **loader_kw), \
           DataLoader(test_ds, shuffle=False, **loader_kw)

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

def run_single(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"  [seed={seed}] Loading data...", flush=True)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders()
    print(f"  [seed={seed}] Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)
    
    model = CNNModel(num_classes=2, input_channels=train_ds.get_num_channels(), arch='resnet18', pretrained=False).to(DEVICE)
    if hasattr(torch, 'compile'):
        try: model = torch.compile(model, mode='reduce-overhead'); print(f"  [seed={seed}] Compiled", flush=True)
        except: pass
    
    criterion = nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = GradScaler('cuda')
    
    best_f1, best_state, patience_cnt, history = 0, None, 0, []
    print(f"  [seed={seed}] Training...", flush=True)
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step()
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_m['accuracy'], 'val_f1': val_m['f1']})
        print(f"  [seed={seed}] Epoch {epoch+1}/{CONFIG['epochs']}: loss={train_loss:.4f}, acc={train_acc:.4f}, val_acc={val_m['accuracy']:.4f}, val_f1={val_m['f1']:.4f}", flush=True)
        if val_m['f1'] > best_f1:
            best_f1, best_state, patience_cnt = val_m['f1'], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG['patience']: print(f"  [seed={seed}] Early stop", flush=True); break
    
    if best_state: model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_m = evaluate(model, test_loader, criterion)
    print(f"  [seed={seed}] Test: acc={test_m['accuracy']:.4f}, f1={test_m['f1']:.4f}, auc={test_m['auc']:.4f}", flush=True)
    return {'seed': seed, 'test_metrics': test_m, 'best_val_f1': best_f1, 'epochs_trained': len(history), 'history': history}

def main():
    print("="*60 + f"\n{CONFIG['name']} Experiment\n" + "="*60, flush=True)
    print(f"Start: {datetime.now()}, Device: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    runs = []
    for i in range(CONFIG['num_runs']):
        print(f"\n--- Run {i+1}/{CONFIG['num_runs']} ---", flush=True)
        runs.append(run_single(42 + i * 100))
    
    summary = {}
    for k in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
        vals = [r['test_metrics'][k] for r in runs]
        summary[f'{k}_mean'], summary[f'{k}_std'] = float(np.mean(vals)), float(np.std(vals))
    
    print(f"\n=== {CONFIG['name']} Summary ===", flush=True)
    print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}", flush=True)
    print(f"F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}", flush=True)
    print(f"AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}", flush=True)
    
    result = {'name': CONFIG['name'], 'config': CONFIG, 'runs': runs, 'summary': summary}
    with open(OUTPUT_DIR / f"{CONFIG['name'].replace('-', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"End: {datetime.now()}", flush=True)

if __name__ == '__main__':
    main()
