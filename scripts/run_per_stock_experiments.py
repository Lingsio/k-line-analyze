"""
Per-Stock Training Experiments
每只股票单独训练独立模型，验证个股专用特征学习效果

实验设计：
- Window: 60天 K线数据
- Prediction: 未来5天涨跌
- 每只股票训练独立的CNN模型
- 对比混合训练 vs 单独训练的效果

用法：
    # Pilot: 测试10只股票（约30分钟）
    python scripts/run_per_stock_experiments.py --num_stocks 10

    # 完整实验: 100只股票（约10-15小时，可断点续传）
    python scripts/run_per_stock_experiments.py --num_stocks 100 --parallel 2
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path
import pandas as pd
from collections import defaultdict
import multiprocessing
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

# ============================================================================
# Configuration
# ============================================================================

# 硬件配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = min(4, multiprocessing.cpu_count())  # Per-stock用较少worker
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()

# Per-stock训练配置（相比全局训练更保守）
CONFIG = {
    'window_size': 60,           # 60天窗口（原来是20天）
    'prediction_horizon': 5,     # 预测未来5天
    'img_size': (128, 128),
    'epochs': 15,                # 减少epoch防止过拟合
    'batch_size': 32,            # 单股数据量小，用小batch
    'lr': 2e-4,                  # 较小学习率
    'weight_decay': 1e-4,        # 更强正则化
    'patience': 3,               # 更激进的早停
    'num_runs': 1,               # 先只跑1次（可改为3次）
    'norm_method': 'minmax',     # 简单归一化
    'use_clahe': False,
    'output_channels': 'rgb',
    'augment_prob': 0.2,         # 轻度数据增强
}

# 数据目录
DATA_DIRS = [
    str(PROJECT_ROOT / 'data' / 'raw')
]

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'per_stock_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def get_stock_list(data_dirs, num_stocks=None):
    """获取股票列表"""
    all_files = []
    for data_dir in data_dirs:
        csv_files = sorted(Path(data_dir).glob('*.csv'))
        all_files.extend(csv_files)

    # 过滤掉太小的文件（数据不足）
    min_samples = CONFIG['window_size'] + CONFIG['prediction_horizon'] + 100
    valid_files = []
    for f in all_files:
        df = pd.read_csv(f)
        if len(df) >= min_samples:
            valid_files.append(f)

    print(f"Found {len(valid_files)} valid stocks (>= {min_samples} days data)")

    # 如果指定数量，随机采样
    if num_stocks and num_stocks < len(valid_files):
        np.random.seed(42)
        valid_files = np.random.choice(valid_files, num_stocks, replace=False).tolist()
        print(f"Selected {num_stocks} stocks for experiment")

    return valid_files


def create_single_stock_dataset(stock_file, mode='train'):
    """为单只股票创建数据集"""
    dataset = StockDataset(
        data_dir=[str(stock_file.parent)],  # 只用这只股票所在目录
        mode=mode,
        window_size=CONFIG['window_size'],
        prediction_horizon=CONFIG['prediction_horizon'],
        img_size=CONFIG['img_size'],
        norm_method=CONFIG['norm_method'],
        use_clahe=CONFIG['use_clahe'],
        output_channels=CONFIG['output_channels'],
        augment_prob=CONFIG['augment_prob'] if mode == 'train' else 0.0,
        label_threshold=0.005,
    )

    # 过滤出只属于这只股票的样本
    stock_name = stock_file.stem
    stock_indices = [i for i, sample in enumerate(dataset.samples)
                     if sample['ticker'] == stock_name]

    if len(stock_indices) == 0:
        return None, 0

    subset = Subset(dataset, stock_indices)
    return subset, len(stock_indices)


def train_single_stock(stock_file, seed=42):
    """训练单只股票的模型"""
    stock_name = stock_file.stem
    torch.manual_seed(seed)
    np.random.seed(seed)

    result = {
        'stock_name': stock_name,
        'stock_file': str(stock_file),
        'seed': seed,
        'success': False,
        'error': None,
    }

    try:
        # 创建数据集
        train_subset, train_size = create_single_stock_dataset(stock_file, 'train')
        val_subset, val_size = create_single_stock_dataset(stock_file, 'val')
        test_subset, test_size = create_single_stock_dataset(stock_file, 'test')

        # 检查数据量
        if train_size < 50 or val_size < 20 or test_size < 20:
            result['error'] = f"Insufficient data: train={train_size}, val={val_size}, test={test_size}"
            return result

        result['data_size'] = {
            'train': train_size,
            'val': val_size,
            'test': test_size,
        }

        # 创建DataLoader
        loader_kwargs = {
            'batch_size': min(CONFIG['batch_size'], train_size // 4),  # 自适应batch size
            'num_workers': NUM_WORKERS,
            'pin_memory': PIN_MEMORY,
        }

        train_loader = DataLoader(train_subset, shuffle=True, drop_last=False, **loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

        # 创建模型
        model = CNNModel(
            num_classes=2,
            input_channels=3,  # RGB
            arch='resnet18',
            pretrained=False
        ).to(DEVICE)

        # 计算类别权重
        train_labels = []
        for batch in train_loader:
            _, _, labels = batch
            train_labels.extend(labels.numpy())

        unique, counts = np.unique(train_labels, return_counts=True)
        class_weights = torch.FloatTensor([counts.sum() / (len(unique) * c) for c in counts]).to(DEVICE)

        # 训练设置
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        scaler = GradScaler() if USE_AMP else None

        # 训练循环
        best_val_f1 = 0
        best_state = None
        patience_counter = 0
        history = []

        for epoch in range(CONFIG['epochs']):
            # Train
            model.train()
            train_loss = 0
            train_preds, train_labels_epoch = [], []

            for batch in train_loader:
                imgs, _, labels = batch
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    continue

                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                train_loss += loss.item()
                train_preds.extend(outputs.argmax(1).detach().cpu().numpy())
                train_labels_epoch.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels_epoch, train_preds) if train_labels_epoch else 0

            # Validation
            model.eval()
            val_preds, val_labels, val_probs = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    imgs, _, labels = batch
                    imgs = imgs.to(DEVICE, non_blocking=True)

                    with autocast(device_type='cuda', enabled=USE_AMP):
                        outputs = model(imgs)

                    probs = torch.softmax(outputs.float(), dim=1)
                    val_preds.extend(outputs.argmax(1).cpu().numpy())
                    val_labels.extend(labels.numpy())
                    val_probs.extend(probs[:, 1].cpu().numpy())

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            scheduler.step()

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_f1': val_f1,
            })

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    break

        # 测试
        if best_state:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        model.eval()
        test_preds, test_labels, test_probs = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                imgs, _, labels = batch
                imgs = imgs.to(DEVICE, non_blocking=True)

                with autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = model(imgs)

                probs = torch.softmax(outputs.float(), dim=1)
                test_preds.extend(outputs.argmax(1).cpu().numpy())
                test_labels.extend(labels.numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())

        # 计算指标
        test_metrics = {
            'accuracy': float(accuracy_score(test_labels, test_preds)),
            'f1': float(f1_score(test_labels, test_preds, average='weighted')),
            'precision': float(precision_score(test_labels, test_preds, average='weighted', zero_division=0)),
            'recall': float(recall_score(test_labels, test_preds, average='weighted', zero_division=0)),
        }

        try:
            test_metrics['auc'] = float(roc_auc_score(test_labels, test_probs))
        except:
            test_metrics['auc'] = 0.5

        result['test_metrics'] = test_metrics
        result['best_val_f1'] = float(best_val_f1)
        result['epochs_trained'] = len(history)
        result['history'] = history
        result['success'] = True

        # 保存模型（可选）
        # model_path = OUTPUT_DIR / 'models' / f'{stock_name}_seed{seed}.pt'
        # model_path.parent.mkdir(exist_ok=True)
        # torch.save(best_state, model_path)

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


# ============================================================================
# Main Experiment
# ============================================================================

def run_per_stock_experiment(num_stocks=10, parallel=1, resume=False):
    """
    运行per-stock实验

    Args:
        num_stocks: 股票数量
        parallel: 并行数量（目前未实现，预留）
        resume: 是否从断点续传
    """
    print("="*70)
    print("Per-Stock Training Experiment")
    print("="*70)
    print(f"Config: {CONFIG}")
    print(f"Device: {DEVICE}")
    print(f"Stocks to train: {num_stocks}")
    print(f"Runs per stock: {CONFIG['num_runs']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # 获取股票列表
    stock_files = get_stock_list(DATA_DIRS, num_stocks)

    # 断点续传逻辑
    results_file = OUTPUT_DIR / f'per_stock_results_{num_stocks}stocks.json'
    completed_stocks = set()

    if resume and results_file.exists():
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        completed_stocks = set(r['stock_name'] for r in existing_results['individual_results']
                              if r.get('success'))
        print(f"Resuming: {len(completed_stocks)} stocks already completed")

    # 训练每只股票
    all_results = []
    successful_results = []
    failed_stocks = []

    start_time = datetime.now()

    for i, stock_file in enumerate(tqdm(stock_files, desc="Training stocks")):
        stock_name = stock_file.stem

        # 跳过已完成的
        if stock_name in completed_stocks:
            print(f"[{i+1}/{len(stock_files)}] Skipping {stock_name} (already done)")
            continue

        print(f"\n[{i+1}/{len(stock_files)}] Training {stock_name}...")

        # 训练（可以跑多次seeds）
        stock_results = []
        for run_id in range(CONFIG['num_runs']):
            seed = 42 + run_id * 100
            result = train_single_stock(stock_file, seed=seed)
            stock_results.append(result)

        # 汇总这只股票的结果
        if any(r['success'] for r in stock_results):
            # 取多次运行的平均
            metrics_list = [r['test_metrics'] for r in stock_results if r['success']]
            avg_metrics = {
                k: float(np.mean([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }

            summary = {
                'stock_name': stock_name,
                'stock_file': str(stock_file),
                'success': True,
                'num_successful_runs': len(metrics_list),
                'avg_metrics': avg_metrics,
                'runs': stock_results,
            }

            successful_results.append(summary)

            print(f"  ✓ Success: acc={avg_metrics['accuracy']:.4f}, "
                  f"f1={avg_metrics['f1']:.4f}, auc={avg_metrics['auc']:.4f}")
        else:
            summary = {
                'stock_name': stock_name,
                'stock_file': str(stock_file),
                'success': False,
                'error': stock_results[0].get('error', 'Unknown error'),
            }
            failed_stocks.append(summary)
            print(f"  ✗ Failed: {summary['error']}")

        all_results.append(summary)

        # 每10只股票保存一次（断点续传）
        if (i + 1) % 10 == 0 or (i + 1) == len(stock_files):
            save_intermediate_results(all_results, successful_results, failed_stocks,
                                     start_time, results_file)

    # 最终汇总
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    if successful_results:
        # 计算所有股票的平均指标
        all_metrics = [r['avg_metrics'] for r in successful_results]
        overall_avg = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0].keys()
        }
        overall_std = {
            k: float(np.std([m[k] for m in all_metrics]))
            for k in all_metrics[0].keys()
        }

        print(f"Successful stocks: {len(successful_results)} / {len(stock_files)}")
        print(f"Failed stocks: {len(failed_stocks)}")
        print()
        print("Average performance across all stocks:")
        print(f"  Accuracy: {overall_avg['accuracy']:.4f} ± {overall_std['accuracy']:.4f}")
        print(f"  F1 Score: {overall_avg['f1']:.4f} ± {overall_std['f1']:.4f}")
        print(f"  AUC:      {overall_avg['auc']:.4f} ± {overall_std['auc']:.4f}")
        print()
        print("Best performing stocks:")
        top_5 = sorted(successful_results, key=lambda x: x['avg_metrics']['f1'], reverse=True)[:5]
        for r in top_5:
            print(f"  {r['stock_name']}: f1={r['avg_metrics']['f1']:.4f}, "
                  f"acc={r['avg_metrics']['accuracy']:.4f}")

        print()
        print("Worst performing stocks:")
        bottom_5 = sorted(successful_results, key=lambda x: x['avg_metrics']['f1'])[:5]
        for r in bottom_5:
            print(f"  {r['stock_name']}: f1={r['avg_metrics']['f1']:.4f}, "
                  f"acc={r['avg_metrics']['accuracy']:.4f}")

        # 保存最终结果
        final_results = {
            'config': CONFIG,
            'num_stocks': len(stock_files),
            'num_successful': len(successful_results),
            'num_failed': len(failed_stocks),
            'overall_avg': overall_avg,
            'overall_std': overall_std,
            'individual_results': all_results,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_time_seconds': (datetime.now() - start_time).total_seconds(),
        }

        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        print("="*70)

        return final_results
    else:
        print("No successful results!")
        return None


def save_intermediate_results(all_results, successful_results, failed_stocks,
                              start_time, output_file):
    """保存中间结果（用于断点续传）"""
    intermediate = {
        'config': CONFIG,
        'num_completed': len(all_results),
        'num_successful': len(successful_results),
        'num_failed': len(failed_stocks),
        'individual_results': all_results,
        'start_time': start_time.isoformat(),
        'last_update': datetime.now().isoformat(),
    }

    with open(output_file, 'w') as f:
        json.dump(intermediate, f, indent=2)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Per-Stock Training Experiment')
    parser.add_argument('--num_stocks', type=int, default=10,
                       help='Number of stocks to train (default: 10 for pilot)')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel training jobs (not implemented yet)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous checkpoint')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per stock (default: 1)')

    args = parser.parse_args()

    # 更新配置
    CONFIG['num_runs'] = args.runs

    # 运行实验
    run_per_stock_experiment(
        num_stocks=args.num_stocks,
        parallel=args.parallel,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
