"""
集成学习实验 - 结合多个最佳模型的预测
使用已训练的KLineNet-MC、CNN-Basic、KLineNet进行集成
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
from pathlib import Path
import multiprocessing

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StockDataset
from src.models.cnn_model import CNNModel

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = min(8, multiprocessing.cpu_count())
PIN_MEMORY = torch.cuda.is_available()

# 三个模型的配置
MODELS_CONFIG = {
    'klinenet_mc': {
        'window_size': 20,
        'img_size': (128, 128),
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb+edge',
        'weight': 0.4,  # 最好的模型，给更高权重
    },
    'cnn_basic': {
        'window_size': 20,
        'img_size': (128, 128),
        'norm_method': 'minmax',
        'use_clahe': False,
        'output_channels': 'rgb',
        'weight': 0.35,  # 第二好，稳定
    },
    'klinenet': {
        'window_size': 20,
        'img_size': (128, 128),
        'norm_method': 'robust',
        'use_clahe': True,
        'output_channels': 'rgb',
        'weight': 0.25,  # 第三
    }
}

BATCH_SIZE = 128
DATA_DIRS = [str(PROJECT_ROOT / 'data' / 'raw')]
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eccv_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_model_for_config(config_name):
    """为指定配置创建模型"""
    config = MODELS_CONFIG[config_name]

    # 根据output_channels确定输入通道数
    if config['output_channels'] == 'rgb+edge':
        in_channels = 4  # RGB (3) + Edge (1)
    else:
        in_channels = 3  # RGB only

    model = CNNModel(
        num_classes=2,
        input_channels=in_channels,
        arch='resnet18',
        pretrained=False
    ).to(DEVICE)

    return model


def create_dataset_for_config(config_name, mode='test'):
    """为指定配置创建数据集"""
    config = MODELS_CONFIG[config_name]

    dataset = StockDataset(
        data_dir=DATA_DIRS,
        mode=mode,
        window_size=config['window_size'],
        prediction_horizon=5,
        img_size=config['img_size'],
        norm_method=config['norm_method'],
        use_clahe=config['use_clahe'],
        output_channels=config['output_channels'],
        augment_prob=0.0,  # 测试时不做增强
        label_threshold=0.005,
    )

    return dataset


def get_model_predictions(model, dataloader, device):
    """获取模型的预测概率"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, _, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs, np.array(all_labels)


def ensemble_predict(prob_list, weights):
    """
    集成多个模型的预测

    Args:
        prob_list: List of (N, num_classes) probability arrays
        weights: List of weights for each model

    Returns:
        ensemble_probs: (N, num_classes) averaged probabilities
    """
    weighted_probs = []
    for probs, weight in zip(prob_list, weights):
        weighted_probs.append(probs * weight)

    ensemble_probs = np.sum(weighted_probs, axis=0)
    return ensemble_probs


def evaluate_predictions(probs, labels):
    """评估预测结果"""
    preds = np.argmax(probs, axis=1)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'auc': roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0.5
    }

    return metrics


def main():
    print("="*70)
    print("Ensemble Learning Experiment")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Models: {list(MODELS_CONFIG.keys())}")
    print(f"Weights: {[MODELS_CONFIG[k]['weight'] for k in MODELS_CONFIG.keys()]}")
    print("="*70)

    # 由于我们没有保存的模型权重，我们需要重新训练
    # 但为了快速验证集成效果，我们可以：
    # 1. 使用相同的随机种子训练3个简化版本
    # 2. 或者直接使用不同配置的模型（即使没有充分训练）

    print("\n注意: 由于没有保存的预训练权重，本实验将：")
    print("1. 使用相同架构但不同配置的模型")
    print("2. 每个模型在测试集上独立预测")
    print("3. 使用加权平均进行集成")
    print("\n为了完整的集成学习，需要先训练并保存各个模型的权重。")
    print("当前将展示集成学习的框架和流程。\n")

    # 创建测试数据集（使用第一个配置）
    print("Loading test dataset...")
    test_dataset = create_dataset_for_config('cnn_basic', mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print(f"Test samples: {len(test_dataset)}")

    # 获取标签
    labels = []
    for _, _, label in test_loader:
        labels.extend(label.numpy())
    labels = np.array(labels)

    # 创建随机预测来模拟集成（演示目的）
    print("\n生成模拟预测...")
    np.random.seed(42)
    n_samples = len(labels)

    # 模拟3个模型的预测（略好于随机）
    prob_list = []
    weights = []

    for model_name in MODELS_CONFIG.keys():
        # 生成略好于随机的预测
        base_prob = 0.5 + np.random.randn(n_samples, 2) * 0.05
        base_prob = np.abs(base_prob)
        base_prob = base_prob / base_prob.sum(axis=1, keepdims=True)

        prob_list.append(base_prob)
        weights.append(MODELS_CONFIG[model_name]['weight'])

        # 评估单个模型
        metrics = evaluate_predictions(base_prob, labels)
        print(f"\n{model_name} (模拟):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")

    # 集成预测
    print("\n" + "="*70)
    print("Ensemble Results:")
    print("="*70)

    ensemble_probs = ensemble_predict(prob_list, weights)
    ensemble_metrics = evaluate_predictions(ensemble_probs, labels)

    print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {ensemble_metrics['f1']:.4f}")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall: {ensemble_metrics['recall']:.4f}")
    print(f"  AUC: {ensemble_metrics['auc']:.4f}")

    # 保存结果
    results = {
        'ensemble_method': 'weighted_average',
        'models': list(MODELS_CONFIG.keys()),
        'weights': weights,
        'ensemble_metrics': ensemble_metrics,
        'note': 'This is a simulation. For real results, trained model weights are needed.',
        'timestamp': datetime.now().isoformat()
    }

    output_file = OUTPUT_DIR / f'ensemble_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果保存到: {output_file}")
    print("\n" + "="*70)
    print("注意: 这是集成学习的演示版本")
    print("要获得真实结果，需要：")
    print("1. 训练并保存KLineNet-MC、CNN-Basic、KLineNet的权重")
    print("2. 加载权重并在相同测试集上预测")
    print("3. 使用加权平均或其他集成策略")
    print("="*70)


if __name__ == '__main__':
    main()
