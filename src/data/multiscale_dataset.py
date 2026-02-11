"""
Multi-Scale Dataset for K-line Pattern Recognition

Generates multiple time-scale images (5, 10, 20 days) for each sample.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from .image_generator import ImageGenerator


class MultiScaleStockDataset(Dataset):
    """
    Dataset that generates multi-scale K-line images (5, 10, 20 days).
    
    Each sample contains:
    - 5-day window image
    - 10-day window image  
    - 20-day window image
    - Target label (future return)
    """
    
    def __init__(
        self,
        data_dir,
        scales=[5, 10, 20],
        prediction_horizon=5,
        img_size=(256, 256),
        mode='train',
        split_ratio=(0.7, 0.15, 0.15),
        label_threshold='dynamic',
        limit=None,
        chart_type='ohlc',
        augment_prob=0.3,
        use_gaf=False,
    ):
        """
        Args:
            data_dir: Directory containing stock data files
            scales: List of window sizes (default: [5, 10, 20])
            prediction_horizon: Days ahead to predict
            img_size: Image size for all scales (will be scaled appropriately)
            mode: 'train', 'val', or 'test'
            split_ratio: (train, val, test) split ratios
            label_threshold: 'dynamic' or float
            limit: Limit number of stocks to load
            chart_type: 'ohlc', 'candle', 'gaf', 'hybrid'
            augment_prob: Augmentation probability
            use_gaf: Whether to use GAF encoding
        """
        self.data_dir = data_dir
        self.scales = sorted(scales)  # [5, 10, 20]
        self.max_scale = max(scales)  # 20
        self.prediction_horizon = prediction_horizon
        self.img_size = img_size
        self.mode = mode
        self.label_threshold = label_threshold
        self.limit = limit
        self.chart_type = chart_type
        self.augment_prob = augment_prob if mode == 'train' else 0.0
        self.use_gaf = use_gaf
        
        # Image generators for each scale
        self.img_gens = {}
        for scale in scales:
            # Scale image size proportionally
            scale_factor = scale / 10  # 10-day is base size
            scale_size = (int(img_size[0] * scale_factor), int(img_size[1] * scale_factor))
            self.img_gens[scale] = ImageGenerator(
                img_size=scale_size,
                norm_method='robust',
                include_volume=True
            )
        
        self.samples = []
        self.data_cache = {}
        self.volatility_cache = {}
        
        self._load_data(split_ratio)
    
    def _compute_stock_volatility(self, df):
        """Compute rolling volatility for dynamic threshold."""
        returns = df['Close'].pct_change().dropna()
        if len(returns) < 20:
            return returns.std() if len(returns) > 1 else 0.02
        return returns.rolling(20).std().median()
    
    def _get_dynamic_threshold(self, ticker):
        """Get dynamic threshold based on volatility."""
        vol = self.volatility_cache.get(ticker, 0.02)
        threshold = 0.5 * vol * np.sqrt(self.prediction_horizon)
        return max(0.003, min(0.05, threshold))
    
    def _load_data(self, split_ratio):
        """Load data and create samples."""
        if isinstance(self.data_dir, (list, tuple)):
            data_dirs = self.data_dir
        else:
            data_dirs = [self.data_dir]
        
        all_files = []
        for data_dir in data_dirs:
            csv_files = [(f, data_dir) for f in os.listdir(data_dir) if f.endswith(('.csv', '.parquet'))]
            all_files.extend(csv_files)
        
        if self.limit:
            all_files = all_files[:self.limit]
        
        for file, data_dir in all_files:
            ticker = file.replace('.csv', '').replace('.parquet', '')
            try:
                file_path = os.path.join(data_dir, file)
                if file.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                else:
                    df = pd.read_csv(file_path)
                
                # Standardize column names
                df.columns = [col.title() if col.lower() in ['open', 'high', 'low', 'close', 'volume', 'date'] 
                             else col for col in df.columns]
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date').reset_index(drop=True)
                
                req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in req_cols):
                    continue
                
                df.dropna(subset=req_cols, inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                # Ensure numeric
                for col in req_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(subset=req_cols, inplace=True)
                
                self.data_cache[ticker] = df
                self.volatility_cache[ticker] = self._compute_stock_volatility(df)
                
                # Need max_scale + prediction_horizon days
                total_len = len(df)
                min_required = self.max_scale + self.prediction_horizon
                if total_len < min_required:
                    continue
                
                # Time-based split
                train_end = int(total_len * split_ratio[0])
                val_end = int(total_len * (split_ratio[0] + split_ratio[1]))
                
                if self.mode == 'train':
                    indices = range(0, train_end - min_required)
                elif self.mode == 'val':
                    indices = range(train_end, val_end - min_required)
                else:  # test
                    indices = range(val_end, total_len - min_required)
                
                for i in indices:
                    # Future price for label
                    future_idx = i + self.max_scale + self.prediction_horizon - 1
                    current_idx = i + self.max_scale - 1
                    
                    current_close = df.iloc[current_idx]['Close']
                    future_close = df.iloc[future_idx]['Close']
                    ret = (future_close - current_close) / current_close
                    
                    # Dynamic threshold
                    if self.label_threshold == 'dynamic':
                        threshold = self._get_dynamic_threshold(ticker)
                    else:
                        threshold = float(self.label_threshold)
                    
                    label = 1 if ret > 0 else 0
                    
                    self.samples.append({
                        'ticker': ticker,
                        'start_idx': i,
                        'label': label,
                        'raw_return': ret,
                        'threshold': threshold,
                    })
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def _generate_scale_image(self, sample, scale, apply_aug=False):
        """Generate image for a specific scale."""
        df = self.data_cache[sample['ticker']]
        
        # For multi-scale, the start_idx is the same for all scales
        # We extract different window lengths from the same starting point
        start = sample['start_idx'] + self.max_scale - scale
        end = start + scale
        
        window_df = df.iloc[start:end]
        
        opens = window_df['Open'].values.astype(np.float64)
        highs = window_df['High'].values.astype(np.float64)
        lows = window_df['Low'].values.astype(np.float64)
        closes = window_df['Close'].values.astype(np.float64)
        vols = window_df['Volume'].values.astype(np.float64)
        
        img_gen = self.img_gens[scale]
        
        # Apply augmentation if training
        if apply_aug and np.random.random() < self.augment_prob:
            # Apply data augmentation
            aug_type = np.random.choice(['noise', 'scale', 'time_warp'])
            opens, highs, lows, closes, vols = self._apply_augmentation(
                opens, highs, lows, closes, vols, aug_type
            )
        
        # Generate image based on chart type
        if self.chart_type == 'ohlc':
            img = img_gen.draw_ohlc_bars(opens, highs, lows, closes, vols, 
                                         grayscale=False, include_volume=True)
        elif self.chart_type == 'gaf':
            img = img_gen.create_gaf_ohlc(opens, highs, lows, closes, vols,
                                          method='gasf', combine_channels=False)
        elif self.chart_type == 'hybrid':
            img = img_gen.draw_hybrid(opens, highs, lows, closes, vols, mode='ohlc_gaf')
        else:  # candle
            img = img_gen.fast_cv2_draw(opens, highs, lows, closes, vols, grayscale=False)
        
        # Ensure 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] > 3:
            img = img[:, :, :3]  # Take first 3 channels
        
        return img
    
    def _apply_augmentation(self, opens, highs, lows, closes, volumes, aug_type):
        """Apply data augmentation."""
        o, h, l, c, v = opens.copy(), highs.copy(), lows.copy(), closes.copy(), volumes.copy()
        
        if aug_type == 'noise':
            noise_level = 0.01
            o *= (1 + np.random.normal(0, noise_level, len(o)))
            h *= (1 + np.random.normal(0, noise_level, len(h)))
            l *= (1 + np.random.normal(0, noise_level, len(l)))
            c *= (1 + np.random.normal(0, noise_level, len(c)))
            h = np.maximum.reduce([o, h, l, c])
            l = np.minimum.reduce([o, h, l, c])
            
        elif aug_type == 'scale':
            scale = np.random.uniform(0.98, 1.02)
            mean_p = (o.mean() + c.mean()) / 2
            o = (o - mean_p) * scale + mean_p
            h = (h - mean_p) * scale + mean_p
            l = (l - mean_p) * scale + mean_p
            c = (c - mean_p) * scale + mean_p
            
        elif aug_type == 'time_warp':
            n = len(o)
            warp = np.random.uniform(0.9, 1.1)
            new_n = max(5, int(n * warp))
            
            def interpolate(arr):
                return np.interp(
                    np.linspace(0, new_n-1, n),
                    np.arange(new_n),
                    np.interp(np.linspace(0, n-1, new_n), np.arange(n), arr)
                )
            
            o = interpolate(o)
            h = interpolate(h)
            l = interpolate(l)
            c = interpolate(c)
            v = interpolate(v.astype(float)).astype(int)
        
        return o, h, l, c, v
    
    def __getitem__(self, idx):
        """Get multi-scale sample."""
        sample = self.samples[idx]
        
        # Generate images for each scale
        images = {}
        for scale in self.scales:
            img = self._generate_scale_image(sample, scale, apply_aug=(self.mode == 'train'))
            # Convert to tensor (C, H, W)
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            images[f'{scale}d'] = img_tensor
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return images, label
    
    def get_class_weights(self):
        """Compute class weights for imbalance."""
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels, minlength=2)
        total = len(labels)
        weights = total / (2 * class_counts + 1e-8)
        return torch.tensor(weights, dtype=torch.float32)


def collate_multiscale(batch):
    """Collate function for multi-scale batches."""
    images_dict = {}
    labels = []
    
    # Initialize dict for each scale
    for key in batch[0][0].keys():
        images_dict[key] = []
    
    for images, label in batch:
        for key, img in images.items():
            images_dict[key].append(img)
        labels.append(label)
    
    # Stack tensors
    for key in images_dict:
        images_dict[key] = torch.stack(images_dict[key])
    
    labels = torch.stack(labels)
    
    return images_dict, labels
