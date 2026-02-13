"""
High-Resolution Dataset for H20 training.
Supports up to 512x512 resolution with anti-aliasing.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from .sector_config import get_stock_sector, US_STOCK_GROUPS


class HighResStockDataset(Dataset):
    """
    High-resolution stock dataset for H20 training.
    
    Args:
        data_dir: Directory containing stock parquet files
        years: List of years to include
        window_size: Days to lookback (20 or 60)
        prediction_horizon: Days to predict ahead
        sector: Sector name or None for all
        resolution: 'standard', 'high', 'ultra', 'h20_max'
        use_volume: Whether to include volume subplot
        normalize: Price normalization method
    """
    
    # Resolution configurations (height, width)
    RESOLUTIONS = {
        'standard': {
            20: (64, 60),
            60: (96, 180),
        },
        'high': {  # 2x
            20: (128, 120),
            60: (192, 360),
        },
        'ultra': {  # 4x
            20: (256, 240),
            60: (384, 720),
        },
        'h20_max': {  # Max for H20
            20: (512, 480),
            60: (512, 960),
        }
    }
    
    # Bar widths per resolution
    BAR_WIDTHS = {
        'standard': 3,
        'high': 6,
        'ultra': 12,
        'h20_max': 24,
    }
    
    def __init__(
        self,
        data_dir,
        years,
        window_size=20,
        prediction_horizon=5,
        sector=None,
        resolution='high',
        use_volume=True,
        normalize='first_close',  # 'first_close', 'minmax', 'zscore'
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.years = years if isinstance(years, list) else [years]
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.sector = sector
        self.resolution = resolution
        self.use_volume = use_volume
        self.normalize = normalize
        self.transform = transform
        
        self.img_size = self.RESOLUTIONS[resolution][window_size]
        self.bar_width = self.BAR_WIDTHS[resolution]
        
        # Load samples
        self.samples = self._load_samples()
        
        print(f"HighResStockDataset ({resolution}):")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Years: {self.years}")
        print(f"  Sector: {sector or 'All'}")
        print(f"  Image size: {self.img_size}")
        print(f"  Bar width: {self.bar_width}px")
    
    def _load_samples(self):
        """Load all valid samples."""
        samples = []
        
        if self.sector:
            stock_list = US_STOCK_GROUPS[self.sector]["stocks"]
        else:
            stock_list = [s for group in US_STOCK_GROUPS.values() for s in group["stocks"]]
        
        for ticker in stock_list:
            file_path = self.data_dir / f"{ticker}.parquet"
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_parquet(file_path)
                
                # Handle index
                if df.index.name == 'Date' or 'Date' not in df.columns:
                    df = df.reset_index()
                
                # Standardize columns
                df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume', 'date'] else c for c in df.columns]
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                
                # Filter by years
                df['year'] = df['Date'].dt.year
                df = df[df['year'].isin(self.years)]
                
                if len(df) < self.window_size + self.prediction_horizon + 10:
                    continue
                
                # Generate samples
                for i in range(len(df) - self.window_size - self.prediction_horizon):
                    window_df = df.iloc[i:i + self.window_size]
                    future_df = df.iloc[i + self.window_size:i + self.window_size + self.prediction_horizon]
                    
                    if len(window_df) < self.window_size:
                        continue
                    
                    # Calculate return
                    current_price = window_df['Close'].iloc[-1]
                    future_price = future_df['Close'].iloc[-1]
                    
                    if pd.isna(current_price) or pd.isna(future_price) or current_price == 0:
                        continue
                    
                    ret = (future_price - current_price) / current_price
                    label = 1 if ret > 0 else 0
                    
                    # Check valid data
                    if window_df[['Open', 'High', 'Low', 'Close']].isna().any().any():
                        continue
                    if (window_df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
                        continue
                    
                    samples.append({
                        'ticker': ticker,
                        'date': window_df['Date'].iloc[-1],
                        'year': window_df['year'].iloc[-1],
                        'window_df': window_df,
                        'label': label,
                        'return': ret,
                        'sector': get_stock_sector(ticker),
                    })
                    
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
                continue
        
        return samples
    
    def _generate_highres_image(self, window_df):
        """Generate high-resolution OHLC image with anti-aliasing.
        Vectorized: all coordinate calculations done via NumPy in one pass."""
        h, w = self.img_size
        bar_w = self.bar_width
        
        # Extract data
        opens = window_df['Open'].values.astype(np.float64)
        highs = window_df['High'].values.astype(np.float64)
        lows = window_df['Low'].values.astype(np.float64)
        closes = window_df['Close'].values.astype(np.float64)
        volumes = window_df['Volume'].values.astype(np.float64) if 'Volume' in window_df.columns and self.use_volume else None
        
        # Normalize prices
        if self.normalize == 'first_close':
            first_close = closes[0]
            opens = opens / first_close
            highs = highs / first_close
            lows = lows / first_close
            closes = closes / first_close
        elif self.normalize == 'minmax':
            min_p = min(lows.min(), opens.min(), closes.min())
            max_p = max(highs.max(), opens.max(), closes.max())
            range_p = max_p - min_p if max_p != min_p else 1.0
            opens = (opens - min_p) / range_p
            highs = (highs - min_p) / range_p
            lows = (lows - min_p) / range_p
            closes = (closes - min_p) / range_p
        
        # Calculate dimensions
        if volumes is not None and self.use_volume:
            vol_ratio = 0.15
            vol_height = int(h * vol_ratio)
            gap = max(2, h // 50)
            price_height = h - vol_height - gap
        else:
            vol_height = 0
            gap = 0
            price_height = h
        
        # Supersampling for anti-aliasing
        supersample = 2 if self.resolution in ['ultra', 'h20_max'] else 1
        hh, ww = h * supersample, w * supersample
        bar_ws = bar_w * supersample
        vol_h = vol_height * supersample
        gap_s = gap * supersample
        price_h = price_height * supersample
        
        img = np.zeros((hh, ww), dtype=np.float32)
        
        # Vectorized price scale
        min_p = min(lows.min(), opens.min(), closes.min())
        max_p = max(highs.max(), opens.max(), closes.max())
        if max_p == min_p:
            max_p = min_p + 0.01
        price_range = max_p - min_p
        
        # Vectorized price_to_y
        y_offset = (vol_h + gap_s) if vol_h > 0 else 0
        def vec_price_to_y(prices):
            normalized = (prices - min_p) / price_range
            y = (normalized * (price_h - 1)).astype(np.int32)
            return (price_h - 1 - y + y_offset).astype(np.int32)
        
        n_bars = len(opens)
        
        # Pre-compute all coordinates in one pass
        x_bases = (np.arange(n_bars) * bar_ws).astype(np.int32)
        center_xs = (x_bases + bar_ws // 2).astype(np.int32)
        y_o = vec_price_to_y(opens)
        y_h = vec_price_to_y(highs)
        y_l = vec_price_to_y(lows)
        y_c = vec_price_to_y(closes)
        
        thickness = max(1, supersample)
        tick_len = max(2, bar_ws // 3)
        
        # Draw OHLC bars with pre-computed coords
        for i in range(n_bars):
            xb = int(x_bases[i])
            cx = int(center_xs[i])
            # High-low line
            cv2.line(img, (cx, int(y_h[i])), (cx, int(y_l[i])), 1.0, thickness)
            # Open tick (left)
            cv2.line(img, (xb, int(y_o[i])), (xb + tick_len, int(y_o[i])), 1.0, thickness)
            # Close tick (right)
            cv2.line(img, (cx + tick_len, int(y_c[i])), (cx + tick_len * 2, int(y_c[i])), 1.0, thickness)
        
        # Vectorized volume drawing
        if vol_h > 0 and volumes is not None:
            vol_max = np.nanmax(volumes)
            if vol_max > 0:
                vol_norm = volumes / vol_max
                vol_hs = (vol_norm * vol_h).astype(np.int32)
                vol_tops = vol_h - vol_hs
                intensities = 0.3 + 0.5 * vol_norm
                
                for i in range(n_bars):
                    if np.isnan(volumes[i]) or vol_hs[i] <= 0:
                        continue
                    xb = int(x_bases[i])
                    cv2.rectangle(img, (xb, int(vol_tops[i])), (xb + bar_ws - 1, vol_h), float(intensities[i]), -1)
        
        # Downsample for anti-aliasing if supersampled
        if supersample > 1:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        if img.max() > 0:
            img = img / img.max()
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Generate image
        img = self._generate_highres_image(sample['window_df'])
        
        # Add channel dimension
        img = img[np.newaxis, :, :].astype(np.float32)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return {
            'image': img_tensor,
            'label': label_tensor,
            'ticker': sample['ticker'],
            'date': str(sample['date']),
            'year': sample['year'],
            'sector': sample['sector'],
            'return': sample['return'],
        }
    
    def get_class_distribution(self):
        """Get distribution of labels."""
        labels = [s['label'] for s in self.samples]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        return {'positive': n_pos, 'negative': n_neg, 'pos_ratio': n_pos / len(labels) if labels else 0}
