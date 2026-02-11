import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import cv2
from .image_generator import ImageGenerator
from .quantile_filtering import QuantileThresholdCalculator

class StockDataset(Dataset):
    """
    Enhanced Stock Dataset with improved preprocessing and labeling strategies.
    Supports advanced CV preprocessing: CLAHE, edge detection, HSV, Mixup, CutMix.
    
    V2 Improvements (based on research papers):
    - OHLC bar charts (Xiu et al. 2021)
    - GAF encoding (Chen & Tsai 2020)
    - Shorter window (10 days) + Higher resolution (256x256)
    """
    def __init__(self, data_dir, window_size=10, prediction_horizon=5,
                 img_size=(256, 256), mode='train', split_ratio=(0.7, 0.15, 0.15),
                 use_cache=False, grayscale=False, label_threshold='dynamic',
                 limit=None, norm_method='robust', seq_norm='adaptive',
                 augment_prob=0.3, num_classes=2,
                 # Advanced preprocessing options
                 use_clahe=True, use_blur=False, output_channels='rgb',
                 mixup_prob=0.0, cutmix_prob=0.0, mixup_alpha=0.4, cutmix_alpha=1.0,
                 # New chart options
                 chart_type='ohlc',  # 'candle', 'ohlc', 'gaf', 'hybrid', 'sparse'
                 use_gaf=False,
                 gaf_method='gasf',
                 # Quantile-based flat-sample filtering (TRAIN ONLY to avoid data leakage)
                 quantile_filter=None,
                 # Additional training filter for 3-class (filter extreme neutrals during training)
                 train_filter_threshold=None):
        """
        Args:
            data_dir: Directory containing CSV files.
            window_size: Number of past days to use for prediction (default: 10).
            prediction_horizon: Days ahead to predict return.
            img_size: Image size for CNN (default: 256x256 for 10-day windows).
            mode: 'train', 'val', or 'test'.
            split_ratio: Tuple (train, val, test).
            grayscale: If True, generate 1-channel images.
            label_threshold: 'dynamic' (based on volatility), float, or 'none'.
            limit: Limit number of stocks to load.
            norm_method: 'robust', 'minmax', 'percentile', 'zscore' for image.
            seq_norm: 'adaptive', 'zscore', 'minmax' for sequence data.
            augment_prob: Probability of applying augmentation (train only).
            num_classes: 2 (binary) or 3 (up/flat/down).
            use_clahe: Apply CLAHE contrast enhancement (default: True).
            use_blur: Apply Gaussian blur for noise reduction (default: False).
            output_channels: 'rgb', 'rgb+edge', 'rgb+hsv', 'all', 'gaf' for multi-channel.
            mixup_prob: Probability of applying Mixup augmentation (train only).
            cutmix_prob: Probability of applying CutMix augmentation (train only).
            mixup_alpha: Alpha parameter for Mixup beta distribution.
            cutmix_alpha: Alpha parameter for CutMix beta distribution.
            chart_type: 'candle', 'ohlc', 'gaf', 'hybrid', 'sparse'.
            use_gaf: Whether to use GAF encoding as additional channel.
            gaf_method: 'gasf' or 'gadf' for GAF computation.
            quantile_filter: Float 0.0-0.5, fraction of |return| distribution to
                             filter as "flat". Only applied in train mode (NO leakage to val/test). 
                             None disables.
            train_filter_threshold: For 3-class (num_classes=3), filter training samples with
                             |return| below this threshold. Val/Test keep all samples.
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.grayscale = grayscale
        self.label_threshold = label_threshold
        self.limit = limit
        self.norm_method = norm_method
        self.seq_norm = seq_norm
        self.augment_prob = augment_prob if mode == 'train' else 0.0
        self.num_classes = num_classes
        self.augment_types = ['noise', 'scale', 'time_warp', 'cutout']

        # Advanced preprocessing options
        self.use_clahe = use_clahe
        self.use_blur = use_blur
        self.output_channels = output_channels
        self.mixup_prob = mixup_prob if mode == 'train' else 0.0
        self.cutmix_prob = cutmix_prob if mode == 'train' else 0.0
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.img_size = img_size
        
        # New chart options
        self.chart_type = chart_type
        self.use_gaf = use_gaf
        self.gaf_method = gaf_method

        # Quantile-based filtering (TRAIN ONLY to avoid data leakage)
        self.quantile_filter = quantile_filter
        self.quantile_calculator = None
        if quantile_filter is not None and quantile_filter > 0:
            if mode == 'train':
                self.quantile_calculator = QuantileThresholdCalculator(
                    filter_quantile=quantile_filter,
                    prediction_horizon=prediction_horizon,
                )
            # For val/test: quantile_calculator remains None, so no filtering occurs
        
        # Additional training filter for 3-class (filter extreme neutrals during training)
        self.train_filter_threshold = train_filter_threshold if mode == 'train' else None

        # Enhanced image generator
        self.img_gen = ImageGenerator(
            img_size=img_size,
            norm_method=norm_method,
            include_volume=True
        )

        self.samples = []
        self.data_cache = {}
        self.volatility_cache = {}  # Store per-stock volatility for dynamic thresholds

        self._load_data(split_ratio)

        
    def _compute_stock_volatility(self, df):
        """Compute rolling volatility for dynamic threshold calculation."""
        returns = df['Close'].pct_change().dropna()
        if len(returns) < 20:
            return returns.std() if len(returns) > 1 else 0.02
        # Use 20-day rolling volatility
        return returns.rolling(20).std().median()

    def _get_dynamic_threshold(self, ticker, window_volatility=None):
        """
        Get dynamic threshold based on stock's volatility.
        Returns a threshold that filters ~20% of samples as "flat".
        """
        if ticker in self.volatility_cache:
            vol = self.volatility_cache[ticker]
        else:
            vol = 0.02  # Default 2%

        # Threshold = 0.5 * volatility * sqrt(prediction_horizon)
        # This scales with both volatility and time horizon
        threshold = 0.5 * vol * np.sqrt(self.prediction_horizon)
        return max(0.003, min(0.05, threshold))  # Clamp between 0.3% and 5%

    def _compute_label(self, ret, threshold):
        """
        Compute label based on return and threshold.

        For num_classes=2: binary (up/down)
        For num_classes=3: ternary (up/flat/down) based on threshold
        """
        if self.num_classes == 3:
            if ret > threshold:
                return 2  # Up
            elif ret < -threshold:
                return 0  # Down
            else:
                return 1  # Neutral (flat)
        else:
            # Binary: use threshold if provided, otherwise use 0
            if threshold > 0:
                return 1 if ret > threshold else 0
            return 1 if ret > 0 else 0

    def _load_data(self, split_ratio):
        # Support both single directory and list of directories (multi-market)
        if isinstance(self.data_dir, (list, tuple)):
            data_dirs = self.data_dir
        else:
            data_dirs = [self.data_dir]

        all_files = []
        for data_dir in data_dirs:
            # Support both CSV and Parquet files
            csv_files = [(f, data_dir) for f in os.listdir(data_dir) if f.endswith('.csv')]
            parquet_files = [(f, data_dir) for f in os.listdir(data_dir) if f.endswith('.parquet')]
            all_files.extend(csv_files)
            all_files.extend(parquet_files)

        if self.limit:
            all_files = all_files[:self.limit]

        for file, data_dir in all_files:
            ticker = file.replace('.csv', '').replace('.parquet', '')
            try:
                file_path = os.path.join(data_dir, file)
                if file.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    # Reset index if Date is in index (parquet files often have date as index)
                    if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                else:
                    df = pd.read_csv(file_path)

                # Handle duplicate column names (e.g., from Yahoo Finance)
                if df.columns.duplicated().any():
                    # Keep only first occurrence of each column
                    df = df.loc[:, ~df.columns.duplicated()]

                # Standardize column names to title case (handle both uppercase and lowercase)
                df.columns = [col.title() if col.lower() in ['open', 'high', 'low', 'close', 'volume', 'date'] else col
                             for col in df.columns]

                # Ensure date is sorted
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date').reset_index(drop=True)

                # Check required columns
                req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in req_cols):
                    continue

                # Drop NaNs immediately
                df.dropna(subset=req_cols, inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Ensure numeric types
                for col in req_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(subset=req_cols, inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Store full DF in memory
                self.data_cache[ticker] = df

                # Compute and cache volatility for dynamic threshold
                self.volatility_cache[ticker] = self._compute_stock_volatility(df)

                # Fit quantile filter on training data portion
                if self.quantile_calculator is not None:
                    train_end_idx = int(len(df) * split_ratio[0])
                    train_df = df.iloc[:train_end_idx]
                    if len(train_df) > self.prediction_horizon + 10:
                        self.quantile_calculator.fit(train_df, ticker)

                # Generate valid indices
                total_len = len(df)
                if total_len < self.window_size + self.prediction_horizon:
                    continue

                # Time series split logic
                train_end = int(total_len * split_ratio[0])
                val_end = int(total_len * (split_ratio[0] + split_ratio[1]))

                if self.mode == 'train':
                    indices = range(0, train_end - self.window_size - self.prediction_horizon)
                elif self.mode == 'val':
                    indices = range(train_end, val_end - self.window_size - self.prediction_horizon)
                else:  # test
                    indices = range(val_end, total_len - self.window_size - self.prediction_horizon)

                for i in indices:
                    current_close = df.iloc[i + self.window_size - 1]['Close']
                    future_close = df.iloc[i + self.window_size + self.prediction_horizon - 1]['Close']

                    ret = (future_close - current_close) / current_close

                    # Dynamic or fixed threshold
                    if self.label_threshold == 'dynamic':
                        threshold = self._get_dynamic_threshold(ticker)
                    elif self.label_threshold == 'none' or self.label_threshold is None:
                        threshold = 0.0
                    else:
                        threshold = float(self.label_threshold)

                    # IMPORTANT: Only filter training samples to avoid data leakage
                    # Validation and test sets should keep all samples for unbiased evaluation
                    if self.mode == 'train':
                        # For binary classification: skip neutral/flat samples during training
                        if self.num_classes == 2:
                            # Quantile-based filtering (preferred)
                            if self.quantile_calculator is not None:
                                if self.quantile_calculator.should_filter(ticker, ret):
                                    continue
                            # Legacy threshold filtering: skip if |return| < threshold
                            elif threshold > 0 and abs(ret) < threshold:
                                continue
                        # For 3-class: we can optionally filter only extreme neutrals 
                        # (very close to 0) during training to improve signal quality
                        elif self.num_classes == 3 and hasattr(self, 'train_filter_threshold'):
                            if self.train_filter_threshold and abs(ret) < self.train_filter_threshold:
                                continue

                    label = self._compute_label(ret, threshold)

                    self.samples.append({
                        'ticker': ticker,
                        'start_idx': i,
                        'label': label,
                        'raw_return': ret,
                        'threshold': threshold
                    })

            except Exception as e:
                print(f"Error loading {file}: {e}")

    def __len__(self):
        return len(self.samples)

    def _normalize_sequence(self, prices, volume):
        """
        Apply advanced normalization to sequence data.

        Returns normalized price features and volume.
        """
        epsilon = 1e-8

        if self.seq_norm == 'adaptive':
            # Adaptive normalization based on window statistics
            log_rets = np.log((prices[1:] + epsilon) / (prices[:-1] + epsilon))
            log_rets = np.insert(log_rets, 0, 0.0)

            # Adaptive scaling: divide by window std (robust to different volatility)
            std = np.std(log_rets) + epsilon
            norm_prices = log_rets / (std * 3)  # Scale to roughly [-1, 1]
            norm_prices = np.clip(norm_prices, -1, 1)

        elif self.seq_norm == 'zscore':
            # Z-score normalization
            log_rets = np.log((prices[1:] + epsilon) / (prices[:-1] + epsilon))
            log_rets = np.insert(log_rets, 0, 0.0)

            mean = np.mean(log_rets)
            std = np.std(log_rets) + epsilon
            norm_prices = (log_rets - mean) / std
            norm_prices = np.clip(norm_prices, -3, 3) / 3  # Clip and scale

        elif self.seq_norm == 'minmax':
            # MinMax normalization of raw prices
            min_p = np.min(prices)
            max_p = np.max(prices)
            range_p = max_p - min_p + epsilon
            norm_prices = (prices - min_p) / range_p

        else:  # fallback to simple log return * 10
            log_rets = np.log((prices[1:] + epsilon) / (prices[:-1] + epsilon))
            log_rets = np.insert(log_rets, 0, 0.0)
            norm_prices = log_rets * 10.0

        # Volume normalization: use robust scaling
        vol_median = np.median(volume) + epsilon
        norm_vol = np.clip(volume / (vol_median * 3), 0, 1)

        return norm_prices, norm_vol

    def _get_image_for_sample(self, sample, apply_basic_aug=False):
        """
        Generate image for a single sample.
        Used internally for both normal generation and Mixup/CutMix.
        
        V2: Supports OHLC bars, GAF encoding, and hybrid representations.
        """
        df = self.data_cache[sample['ticker']]
        start = sample['start_idx']
        end = start + self.window_size
        window_df = df.iloc[start:end].copy()

        opens = window_df['Open'].values.astype(np.float64)
        highs = window_df['High'].values.astype(np.float64)
        lows = window_df['Low'].values.astype(np.float64)
        closes = window_df['Close'].values.astype(np.float64)
        vols = window_df['Volume'].values.astype(np.float64)

        # Decide if we apply basic augmentation
        augment_type = None
        if apply_basic_aug and np.random.random() < self.augment_prob:
            augment_type = np.random.choice(self.augment_types)

        if self.chart_type == 'sparse':
            # Sparse grayscale OHLC image, no augmentation
            img_np = self.img_gen.draw_sparse_ohlc(
                opens, highs, lows, closes, vols,
                window_size=self.window_size
            )
            # Return as (H, W, 1) for single-channel handling
            return img_np[:, :, np.newaxis]

        if self.chart_type == 'gaf':
            # Pure GAF representation (Chen & Tsai 2020)
            img_np = self.img_gen.create_gaf_ohlc(
                opens, highs, lows, closes, vols,
                size=self.img_size,
                method=self.gaf_method,
                combine_channels=False
            )
            return img_np
        
        elif self.chart_type == 'ohlc':
            # OHLC bar chart (Xiu et al. 2021)
            if augment_type:
                # For OHLC, apply augmentation to data first
                opens, highs, lows, closes, vols = self._apply_data_augmentation(
                    opens, highs, lows, closes, vols, augment_type
                )
            
            img_np = self.img_gen.draw_ohlc_bars(
                opens, highs, lows, closes, vols,
                size=self.img_size,
                grayscale=self.grayscale,
                include_volume=True
            )
            
            # Apply CLAHE if requested (convert to 3ch first if needed)
            if self.use_clahe and not self.grayscale:
                if len(img_np.shape) == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                img_np = self.img_gen.apply_clahe(img_np)
            
            # Add GAF as additional channel if requested
            if self.use_gaf and not self.grayscale:
                gaf = self.img_gen.create_gaf_ohlc(
                    opens, highs, lows, closes, vols,
                    size=self.img_size,
                    method=self.gaf_method,
                    combine_channels=False
                )
                # Blend OHLC and GAF
                img_np = (0.7 * img_np.astype(float) + 0.3 * gaf.astype(float)).astype(np.uint8)
            
            return img_np
        
        elif self.chart_type == 'hybrid':
            # Hybrid: OHLC bars with GAF features
            img_np = self.img_gen.draw_hybrid(
                opens, highs, lows, closes, vols,
                size=self.img_size,
                mode='ohlc_gaf'
            )
            return img_np
        
        else:
            # Original candlestick chart
            if augment_type:
                img_np = self.img_gen.draw_with_augmentation(
                    opens, highs, lows, closes, vols,
                    augment_type=augment_type,
                    grayscale=self.grayscale
                )
                if self.use_clahe and not self.grayscale:
                    img_np = self.img_gen.apply_clahe(img_np)
                if self.output_channels != 'rgb' and not self.grayscale:
                    if self.output_channels == 'rgb+edge':
                        img_np = self.img_gen.create_multi_channel(img_np, include_edge=True, include_hsv_h=False)
                    elif self.output_channels == 'rgb+hsv':
                        img_np = self.img_gen.create_multi_channel(img_np, include_edge=False, include_hsv_h=True)
                    elif self.output_channels == 'all':
                        img_np = self.img_gen.create_multi_channel(img_np, include_edge=True, include_hsv_h=True)
            else:
                img_np = self.img_gen.draw_enhanced(
                    opens, highs, lows, closes, vols,
                    grayscale=self.grayscale,
                    use_clahe=self.use_clahe,
                    use_blur=self.use_blur,
                    output_channels=self.output_channels if not self.grayscale else 'rgb'
                )
            return img_np
    
    def _apply_data_augmentation(self, opens, highs, lows, closes, volumes, augment_type):
        """Apply augmentation directly to OHLCV data (before drawing)."""
        o, h, l, c, v = opens.copy(), highs.copy(), lows.copy(), closes.copy(), volumes.copy()
        
        if augment_type == 'noise':
            noise_level = 0.01
            o *= (1 + np.random.normal(0, noise_level, len(o)))
            h *= (1 + np.random.normal(0, noise_level, len(h)))
            l *= (1 + np.random.normal(0, noise_level, len(l)))
            c *= (1 + np.random.normal(0, noise_level, len(c)))
            # Ensure OHLC consistency
            h = np.maximum.reduce([o, h, l, c])
            l = np.minimum.reduce([o, h, l, c])
            
        elif augment_type == 'scale':
            scale = np.random.uniform(0.98, 1.02)
            mean_price = (o.mean() + c.mean()) / 2
            o = (o - mean_price) * scale + mean_price
            h = (h - mean_price) * scale + mean_price
            l = (l - mean_price) * scale + mean_price
            c = (c - mean_price) * scale + mean_price
            
        elif augment_type == 'time_warp':
            # Simple time warping
            n = len(o)
            warp = np.random.uniform(0.9, 1.1)
            new_n = max(5, int(n * warp))
            indices = np.linspace(0, n-1, new_n)
            
            def interpolate(arr):
                return np.interp(np.linspace(0, new_n-1, n), np.arange(new_n), 
                               np.interp(indices, np.arange(n), arr))
            
            o = interpolate(o)
            h = interpolate(h)
            l = interpolate(l)
            c = interpolate(c)
            v = interpolate(v.astype(float)).astype(int)
        
        return o, h, l, c, v

    def __getitem__(self, idx):
        sample = self.samples[idx]
        df = self.data_cache[sample['ticker']]

        # Slicing
        start = sample['start_idx']
        end = start + self.window_size
        window_df = df.iloc[start:end].copy()

        # Extract raw arrays
        opens = window_df['Open'].values.astype(np.float64)
        highs = window_df['High'].values.astype(np.float64)
        lows = window_df['Low'].values.astype(np.float64)
        closes = window_df['Close'].values.astype(np.float64)
        vols = window_df['Volume'].values.astype(np.float64)

        # --- 1D Features (Sequence) ---
        norm_prices, norm_vol = self._normalize_sequence(closes, vols)
        seq_data = np.column_stack((norm_prices, norm_vol))
        seq_tensor = torch.tensor(seq_data, dtype=torch.float32)

        # --- 2D Features (Image) ---
        apply_basic_aug = self.mode == 'train'
        img_np = self._get_image_for_sample(sample, apply_basic_aug=apply_basic_aug)
        label = sample['label']

        # --- Mixup/CutMix Augmentation (train only) ---
        if self.mode == 'train':
            rand_val = np.random.random()
            if rand_val < self.mixup_prob:
                # Mixup: blend with another random sample
                mix_idx = np.random.randint(len(self.samples))
                mix_sample = self.samples[mix_idx]
                img_np2 = self._get_image_for_sample(mix_sample, apply_basic_aug=False)
                img_np, lam = ImageGenerator.mixup(img_np, img_np2, alpha=self.mixup_alpha)
                # Soft labels not directly supported in CrossEntropy, return mixed label info
                # For now, we pick the dominant label
                label = sample['label'] if lam > 0.5 else mix_sample['label']

            elif rand_val < self.mixup_prob + self.cutmix_prob:
                # CutMix: cut and paste region from another sample
                mix_idx = np.random.randint(len(self.samples))
                mix_sample = self.samples[mix_idx]
                img_np2 = self._get_image_for_sample(mix_sample, apply_basic_aug=False)
                img_np, bbox, lam = ImageGenerator.cutmix(img_np, img_np2, alpha=self.cutmix_alpha)
                # Pick dominant label based on area ratio
                label = sample['label'] if lam > 0.5 else mix_sample['label']

        # HWC -> CHW, Normalize 0-1
        if img_np.ndim == 2:
            # Grayscale (H, W) → (1, H, W)
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0) / 255.0
        else:
            img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # --- Label ---
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, seq_tensor, label_tensor

    def get_class_weights(self):
        """
        Compute class weights for imbalanced datasets.
        Returns tensor of weights inversely proportional to class frequency.
        """
        if not self.samples:
            return torch.ones(self.num_classes, dtype=torch.float32)
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        total = len(labels)
        # Add smoothing to avoid division by zero
        weights = total / (self.num_classes * class_counts + 1e-8)
        # Cap maximum weight to prevent extreme imbalance
        weights = np.clip(weights, 0.1, 10.0)
        return torch.tensor(weights, dtype=torch.float32)

    def get_num_channels(self):
        """
        Get the number of output channels based on configuration.
        Useful for building models with correct input channels.
        """
        if self.chart_type == 'sparse':
            return 1  # Sparse OHLC: single-channel grayscale
        if self.grayscale:
            return 1
        if self.chart_type == 'gaf':
            return 3  # GAF encoded as RGB
        if self.output_channels == 'rgb' or self.output_channels == 'ohlc':
            return 3
        elif self.output_channels == 'rgb+edge':
            return 4  # RGB + Edge
        elif self.output_channels == 'rgb+hsv':
            return 4  # RGB + HSV Hue
        elif self.output_channels == 'all':
            return 5  # RGB + Edge + HSV Hue
        return 3

    def get_stock_full_data(self, ticker):
        """
        Returns the full sorted DataFrame for a specific ticker if it exists.
        """
        if ticker in self.data_cache:
            return self.data_cache[ticker]

        # Support both single directory and list of directories
        if isinstance(self.data_dir, (list, tuple)):
            data_dirs = self.data_dir
        else:
            data_dirs = [self.data_dir]

        # Try loading from all directories, both CSV and parquet
        for data_dir in data_dirs:
            for ext in ['.csv', '.parquet']:
                file_path = os.path.join(data_dir, f"{ticker}{ext}")
                if os.path.exists(file_path):
                    try:
                        if ext == '.parquet':
                            df = pd.read_parquet(file_path)
                            if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                                df = df.reset_index()
                        else:
                            df = pd.read_csv(file_path)
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.sort_values('Date').reset_index(drop=True)
                        return df
                    except:
                        continue
        return None
