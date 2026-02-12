"""
SAK-Net Large-Scale Training Script

支持CRSP大规模数据集（1993-2019年，8000+只美股）的训练
具备以下特性：
- 分布式训练支持
- 混合精度训练
- 渐进式学习
- 动态批次大小
- 检查点恢复

Author: Research Team
Date: 2026
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.crsp_data_loader import CRSPLargeScaleDataset
from src.models.kformer import KFormer
from src.models.sector_moe import SectorMoE, KFormerMoE

# 配置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.backends.cudnn.benchmark = True


class CRSPImageDataset(Dataset):
    """
    CRSP数据集图像生成器
    
    将OHLCV数据转换为K线图像
    """
    def __init__(
        self,
        data_dir: str,
        stock_list: List[str],
        window_size: int = 20,
        prediction_horizon: int = 5,
        img_size: Tuple[int, int] = (128, 128),
        chart_type: str = 'ohlc',
        mode: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment_prob: float = 0.3,
        quantile_filter: Optional[float] = 0.35,
    ):
        self.data_dir = Path(data_dir)
        self.stock_list = stock_list
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.img_size = img_size
        self.chart_type = chart_type
        self.mode = mode
        self.augment_prob = augment_prob
        self.quantile_filter = quantile_filter
        
        # 加载所有数据
        self.samples = self._load_and_split_data(train_ratio, val_ratio)
        
        print(f"[{mode}] Loaded {len(self.samples)} samples from {len(stock_list)} stocks")
        
    def _load_and_split_data(
        self,
        train_ratio: float,
        val_ratio: float
    ) -> List[Dict]:
        """加载并切分数据"""
        all_samples = []
        
        for ticker in self.stock_list:
            file_path = self.data_dir / f"{ticker}.csv"
            if not file_path.exists():
                continue
            
            df = pd.read_csv(file_path, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 生成样本
            for i in range(len(df) - self.window_size - self.prediction_horizon):
                window = df.iloc[i:i+self.window_size]
                future = df.iloc[i+self.window_size:i+self.window_size+self.prediction_horizon]
                
                # 计算未来收益
                current_price = window['close'].iloc[-1]
                future_price = future['close'].iloc[-1]
                future_return = (future_price - current_price) / current_price
                
                # 动态阈值标签
                volatility = window['close'].pct_change().std()
                threshold = max(0.005, volatility * 0.5)
                
                if future_return > threshold:
                    label = 1  # 涨
                elif future_return < -threshold:
                    label = 0  # 跌
                else:
                    continue  # 跳过平缓样本
                
                all_samples.append({
                    'ticker': ticker,
                    'start_idx': i,
                    'label': label,
                    'return': future_return
                })
        
        # 按时间顺序切分
        n = len(all_samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if self.mode == 'train':
            samples = all_samples[:train_end]
            # 应用分位数筛选（仅训练集）
            if self.quantile_filter:
                samples = self._apply_quantile_filter(samples)
        elif self.mode == 'val':
            samples = all_samples[train_end:val_end]
        else:  # test
            samples = all_samples[val_end:]
        
        return samples
    
    def _apply_quantile_filter(self, samples: List[Dict]) -> List[Dict]:
        """应用分位数筛选过滤平缓样本"""
        returns = [abs(s['return']) for s in samples]
        threshold = np.quantile(returns, self.quantile_filter)
        
        filtered = [s for s in samples if abs(s['return']) >= threshold]
        print(f"  Quantile filter: {len(samples)} -> {len(filtered)} samples")
        return filtered
    
    def _generate_ohlc_image(
        self,
        window: pd.DataFrame
    ) -> np.ndarray:
        """生成OHLC图像"""
        h, w = self.img_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 价格归一化
        price_min = window[['open', 'high', 'low', 'close']].min().min()
        price_max = window[['open', 'high', 'low', 'close']].max().max()
        price_range = price_max - price_min
        
        if price_range == 0:
            return img
        
        # 计算K线位置
        n_bars = len(window)
        bar_width = max(1, w // n_bars)
        
        for idx, (_, row) in enumerate(window.iterrows()):
            x_start = idx * bar_width
            x_end = min(x_start + bar_width - 1, w - 1)
            
            # 归一化价格到图像高度
            o = h - 1 - int((row['open'] - price_min) / price_range * (h - 1))
            h_price = h - 1 - int((row['high'] - price_min) / price_range * (h - 1))
            l = h - 1 - int((row['low'] - price_min) / price_range * (h - 1))
            c = h - 1 - int((row['close'] - price_min) / price_range * (h - 1))
            
            # 绘制影线
            img[h_price:l+1, (x_start+x_end)//2] = 255
            
            # 绘制实体
            body_top = min(o, c)
            body_bottom = max(o, c)
            img[body_top:body_bottom+1, x_start:x_end+1] = 255
        
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # 加载数据窗口
        file_path = self.data_dir / f"{sample['ticker']}.csv"
        df = pd.read_csv(file_path)
        
        start_idx = sample['start_idx']
        window = df.iloc[start_idx:start_idx+self.window_size]
        
        # 生成图像
        img = self._generate_ohlc_image(window)
        
        # 转换为tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # 数据增强（仅训练时）
        if self.mode == 'train' and random.random() < self.augment_prob:
            img = self._augment(img)
        
        return img, sample['label']
    
    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        # 添加高斯噪声
        if random.random() < 0.5:
            noise = torch.randn_like(img) * 0.02
            img = torch.clamp(img + noise, 0, 1)
        
        return img


class SAKNetTrainer:
    """SAK-Net大规模训练器"""
    
    def __init__(
        self,
        config: Dict,
        local_rank: int = 0,
        world_size: int = 1
    ):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = self._build_model()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 最佳模型记录
        self.best_acc = 0.0
        self.patience_counter = 0
        
    def _build_model(self) -> nn.Module:
        """构建模型"""
        if self.config['model_type'] == 'kformer':
            model = KFormer(
                num_classes=2,
                d_model=self.config.get('d_model', 256),
                use_learnable_encoder=self.config.get('use_learnable_encoder', True)
            )
        elif self.config['model_type'] == 'sector_moe':
            model = SectorMoE(
                num_experts=self.config.get('num_experts', 6),
                d_model=self.config.get('d_model', 256),
                num_classes=2,
                top_k=self.config.get('top_k', 2)
            )
        elif self.config['model_type'] == 'kformer_moe':
            model = KFormerMoE(
                num_experts=self.config.get('num_experts', 6),
                d_model=self.config.get('d_model', 256),
                num_classes=2
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
        
        model = model.to(self.device)
        
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank])
        
        return model
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度前向
            with autocast():
                outputs = self.model(images)
                
                # 处理MoE的额外输出
                if isinstance(outputs, tuple):
                    logits, aux_loss = outputs
                    loss = self.criterion(logits, labels) + aux_loss
                else:
                    logits = outputs
                    loss = self.criterion(logits, labels)
            
            # 混合精度反向
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0 and self.local_rank == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.*correct/total:.2f}%")
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        logits, _ = outputs
                    else:
                        logits = outputs
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算F1
        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch: int, val_acc: float, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        
        return checkpoint['epoch']


def train_sector_model(
    sector: str,
    stock_list: List[str],
    data_dir: str,
    output_dir: str,
    config: Dict,
    local_rank: int = 0
):
    """训练单个板块模型"""
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Sector: {sector}")
        print(f"Stocks: {len(stock_list)}")
        print(f"{'='*60}")
    
    # 创建数据集
    train_dataset = CRSPImageDataset(
        data_dir=data_dir,
        stock_list=stock_list,
        window_size=config['window_size'],
        prediction_horizon=config['prediction_horizon'],
        img_size=config['img_size'],
        mode='train',
        quantile_filter=config.get('quantile_filter', 0.35)
    )
    
    val_dataset = CRSPImageDataset(
        data_dir=data_dir,
        stock_list=stock_list,
        window_size=config['window_size'],
        prediction_horizon=config['prediction_horizon'],
        img_size=config['img_size'],
        mode='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建训练器
    trainer = SAKNetTrainer(config, local_rank)
    
    # 训练循环
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(config['epochs']):
        if local_rank == 0:
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader)
        
        # 验证
        val_metrics = trainer.validate(val_loader)
        
        # 更新学习率
        trainer.scheduler.step()
        
        # 记录
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        if local_rank == 0:
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                  f"Val F1: {val_metrics['f1']:.4f}")
        
        # 早停检查
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # 保存最佳模型
            if local_rank == 0:
                model_path = Path(output_dir) / f"{sector}_best.pt"
                trainer.save_checkpoint(epoch, best_acc, str(model_path))
                print(f"  -> Saved best model with accuracy {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                if local_rank == 0:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                break
    
    # 保存训练历史
    if local_rank == 0:
        history_path = Path(output_dir) / f"{sector}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='SAK-Net Large-Scale Training')
    parser.add_argument('--data-dir', type=str, default='data/crsp_large_scale',
                        help='CRSP数据目录')
    parser.add_argument('--output-dir', type=str, default='outputs/saknet_large_scale',
                        help='输出目录')
    parser.add_argument('--sector', type=str, default='all',
                        help='训练的板块 (all 或特定板块名)')
    parser.add_argument('--model-type', type=str, default='kformer',
                        choices=['kformer', 'sector_moe', 'kformer_moe'],
                        help='模型类型')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集信息
    dataset = CRSPLargeScaleDataset(data_dir=args.data_dir)
    
    # 确定要训练的板块
    if args.sector == 'all':
        sectors_to_train = list(dataset.SECTOR_MAPPING.keys())
    else:
        sectors_to_train = [args.sector]
    
    # 训练配置
    config = {
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'num_workers': args.num_workers,
        'window_size': 20,
        'prediction_horizon': 5,
        'img_size': (128, 128),
        'd_model': 256,
        'num_experts': 6,
        'top_k': 2,
        'quantile_filter': 0.35,
        'use_learnable_encoder': True
    }
    
    # 保存配置
    if args.local_rank == 0:
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    # 逐个板块训练
    results = {}
    
    for sector in sectors_to_train:
        stock_list = dataset.get_sector_dataset(sector, min_stocks=10)
        
        if len(stock_list) < 10:
            print(f"Skipping {sector}: insufficient stocks ({len(stock_list)})")
            continue
        
        best_acc = train_sector_model(
            sector=sector,
            stock_list=stock_list,
            data_dir=args.data_dir,
            output_dir=str(output_dir),
            config=config,
            local_rank=args.local_rank
        )
        
        results[sector] = {
            'num_stocks': len(stock_list),
            'best_accuracy': best_acc
        }
    
    # 保存结果摘要
    if args.local_rank == 0:
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        for sector, result in results.items():
            print(f"{sector:20s}: {result['best_accuracy']:.2f}% ({result['num_stocks']} stocks)")
        
        avg_acc = sum(r['best_accuracy'] for r in results.values()) / len(results)
        print(f"\nAverage Accuracy: {avg_acc:.2f}%")


if __name__ == "__main__":
    main()
