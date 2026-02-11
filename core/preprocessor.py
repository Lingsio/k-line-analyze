"""
K-Line data preprocessor for normalization and feature extraction.

Key functions:
1. Sliding window creation
2. Min-Max normalization (shape-preserving)
3. GAF (Gramian Angular Field) transformation
4. K-line image generation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from PIL import Image
import io


class KLinePreprocessor:
    """Preprocessor for K-line data normalization and transformation."""

    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize preprocessor.

        Args:
            window_sizes: List of window sizes for multi-scale processing
        """
        self.window_sizes = window_sizes or [20, 60, 120]

    def normalize(
        self, df: pd.DataFrame, method: str = "minmax"
    ) -> pd.DataFrame:
        """
        Normalize OHLCV data to preserve shape patterns.

        Args:
            df: DataFrame with OHLCV columns
            method: 'minmax' or 'zscore'

        Returns:
            Normalized DataFrame
        """
        df = df.copy()

        if method == "minmax":
            # Use the range of all price columns for normalization
            price_cols = ["open", "high", "low", "close"]
            all_prices = df[price_cols].values.flatten()
            price_min = np.min(all_prices)
            price_max = np.max(all_prices)
            price_range = price_max - price_min

            if price_range == 0:
                price_range = 1  # Avoid division by zero

            # Normalize prices to [0, 1]
            for col in price_cols:
                df[col] = (df[col] - price_min) / price_range

            # Normalize volume separately
            if "volume" in df.columns:
                vol_max = df["volume"].max()
                if vol_max > 0:
                    df["volume"] = df["volume"] / vol_max

        elif method == "zscore":
            price_cols = ["open", "high", "low", "close"]
            all_prices = df[price_cols].values.flatten()
            mean = np.mean(all_prices)
            std = np.std(all_prices)

            if std == 0:
                std = 1

            for col in price_cols:
                df[col] = (df[col] - mean) / std

            if "volume" in df.columns:
                vol_mean = df["volume"].mean()
                vol_std = df["volume"].std()
                if vol_std > 0:
                    df["volume"] = (df["volume"] - vol_mean) / vol_std

        elif method == "paper":
            # Methodology from Jiang et al. (2023):
            # 1. Prices are relative to the initial price in the window (or mean)
            # 2. Rescale to fixed range but keep internal proportions
            price_cols = ["open", "high", "low", "close"]
            
            # Using the first open price as the reference point
            ref_price = df["open"].iloc[0]
            if ref_price == 0:
                ref_price = df[price_cols].values.mean()
            if ref_price == 0:
                ref_price = 1.0
                
            for col in price_cols:
                df[col] = df[col] / ref_price
                
            # Bring to [0, 1] range based on the window's own min-max
            all_prices = df[price_cols].values.flatten()
            p_min, p_max = np.min(all_prices), np.max(all_prices)
            p_range = p_max - p_min
            if p_range == 0: p_range = 1
            
            for col in price_cols:
                df[col] = (df[col] - p_min) / p_range
                
            if "volume" in df.columns:
                vol_max = df["volume"].max()
                if vol_max > 0:
                    df["volume"] = df["volume"] / vol_max

        return df

    def create_windows(
        self, df: pd.DataFrame, window_size: int, step: int = 1
    ) -> List[Tuple[pd.DataFrame, int, int]]:
        """
        Create sliding windows from DataFrame.

        Args:
            df: DataFrame with OHLCV data
            window_size: Number of rows per window
            step: Step size between windows

        Returns:
            List of tuples (window_df, start_idx, end_idx)
        """
        windows = []
        n = len(df)

        for i in range(0, n - window_size + 1, step):
            window = df.iloc[i : i + window_size].copy()
            windows.append((window, i, i + window_size - 1))

        return windows

    def to_feature_vector(
        self, df: pd.DataFrame, include_volume: bool = True
    ) -> np.ndarray:
        """
        Convert normalized OHLCV data to a feature vector.

        This creates a simple but effective representation by concatenating
        normalized price and volume data.

        Args:
            df: Normalized DataFrame
            include_volume: Whether to include volume

        Returns:
            1D numpy array (feature vector)
        """
        # Resample to fixed length if needed
        target_len = 60  # Standard length

        if len(df) != target_len:
            df = self._resample_to_length(df, target_len)

        # Extract features
        features = []

        # Price features (OHLC ratios are more invariant)
        close = df["close"].values
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values

        # Close prices (main pattern)
        features.extend(close)

        # Candlestick body ratio (close - open) / (high - low)
        body = close - open_
        wick = high - low
        wick[wick == 0] = 1e-8
        body_ratio = body / wick
        features.extend(body_ratio)

        # Upper/lower shadow ratios
        upper_shadow = (high - np.maximum(open_, close)) / wick
        lower_shadow = (np.minimum(open_, close) - low) / wick
        features.extend(upper_shadow)
        features.extend(lower_shadow)

        # Volume (if included)
        if include_volume and "volume" in df.columns:
            vol = df["volume"].values
            features.extend(vol)

        return np.array(features, dtype=np.float32)

    def _resample_to_length(
        self, df: pd.DataFrame, target_len: int
    ) -> pd.DataFrame:
        """Resample DataFrame to target length using interpolation."""
        from scipy.interpolate import interp1d

        current_len = len(df)
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_len)

        resampled = {}
        for col in df.columns:
            f = interp1d(x_old, df[col].values, kind="linear")
            resampled[col] = f(x_new)

        return pd.DataFrame(resampled)

    def to_gaf_image(
        self, series: np.ndarray, image_size: int = 128, method: str = "gasf"
    ) -> np.ndarray:
        """
        Convert time series to Gramian Angular Field image.

        Args:
            series: 1D time series array
            image_size: Output image size
            method: 'gasf' (Summation) or 'gadf' (Difference)

        Returns:
            2D numpy array (GAF image)
        """
        from app.utils.gaf_transformer import GAFTransformer

        transformer = GAFTransformer(image_size=image_size, method=method)
        return transformer.transform(series)

    def to_kline_image(
        self,
        df: pd.DataFrame,
        image_size: int = 128,
        include_volume: bool = True,
    ) -> np.ndarray:
        """
        Generate a K-line candlestick image.

        Args:
            df: Normalized OHLCV DataFrame
            image_size: Output image size
            include_volume: Whether to include volume bars

        Returns:
            RGB numpy array (image_size x image_size x 3)
        """
        from app.utils.kline_renderer import KLineRenderer

        renderer = KLineRenderer(image_size=image_size)
        return renderer.render(df, include_volume=include_volume)

    def create_multi_channel_image(
        self, df: pd.DataFrame, image_size: int = 128
    ) -> np.ndarray:
        """
        Create a 4-channel image combining K-line and GAF representations.

        Channels:
        - R: Red K-line (bearish candles)
        - G: Green K-line (bullish candles)
        - B: GAF of close prices
        - A: Volume bars

        Args:
            df: Normalized OHLCV DataFrame
            image_size: Output image size

        Returns:
            numpy array (image_size x image_size x 4)
        """
        # K-line image (RGB)
        kline_img = self.to_kline_image(df, image_size, include_volume=False)

        # GAF image (grayscale)
        close_series = df["close"].values
        # Resample close series if it's too short for beautiful GAF
        if len(close_series) < 20: # Ensure minimum length
             # We might need to handle short series, but for now we assume sufficient length or let gaf handle it
             pass
        
        gaf_img = self.to_gaf_image(close_series, image_size)
        gaf_normalized = ((gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8) * 255).astype(np.uint8)

        # Volume bars (grayscale)
        from app.utils.kline_renderer import KLineRenderer
        renderer = KLineRenderer(image_size=image_size)
        vol_img = renderer.render_volume_only(df)

        # Combine channels
        combined = np.zeros((image_size, image_size, 4), dtype=np.uint8)
        combined[:, :, 0] = kline_img[:, :, 0]  # R (bearish)
        combined[:, :, 1] = kline_img[:, :, 1]  # G (bullish)
        combined[:, :, 2] = gaf_normalized       # B (GAF)
        combined[:, :, 3] = vol_img              # A (volume)

        return combined

    def dataframe_to_image(self, df: pd.DataFrame, image_size: int = 128) -> np.ndarray:
        """
        Convert DataFrame to the multi-channel image format expected by the encoder.
        This is an alias for create_multi_channel_image but ensures correct data type (float32 [0,1]).
        
        Args:
            df: Normalized OHLCV DataFrame
            image_size: Output image size
            
        Returns:
            numpy array (Channels x Height x Width) for PyTorch [C, H, W]
            Values normalized to [0, 1]
        """
        # Create HxWxC image first
        img = self.create_multi_channel_image(df, image_size)
        
        # Convert to C x H x W format for PyTorch
        img = np.transpose(img, (2, 0, 1))
        
        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0
        
        return img

    def augment(
        self, df: pd.DataFrame, augmentation_type: str = "noise"
    ) -> pd.DataFrame:
        """
        Apply data augmentation to K-line data.

        Used for creating positive samples in self-supervised learning.

        Args:
            df: Original DataFrame
            augmentation_type: Type of augmentation
                - 'noise': Add random noise
                - 'scale': Slight vertical scaling
                - 'timestretch': Slight horizontal stretching

        Returns:
            Augmented DataFrame
        """
        df = df.copy()
        price_cols = ["open", "high", "low", "close"]

        if augmentation_type == "noise":
            # Add small random noise (σ = 0.01)
            noise = np.random.normal(0, 0.01, (len(df), len(price_cols)))
            df[price_cols] += noise

        elif augmentation_type == "scale":
            # Slight vertical scaling (±5%)
            scale = 1 + np.random.uniform(-0.05, 0.05)
            mean = df[price_cols].values.mean()
            df[price_cols] = (df[price_cols] - mean) * scale + mean

        elif augmentation_type == "timestretch":
            # Time stretch by resampling
            stretch_factor = np.random.uniform(0.9, 1.1)
            new_len = int(len(df) * stretch_factor)
            df = self._resample_to_length(df, new_len)
            # Resample back to original length
            df = self._resample_to_length(df, len(df))

        # Ensure values are still in valid range after augmentation
        df[price_cols] = df[price_cols].clip(0, 1)

        if "volume" in df.columns:
            df["volume"] = df["volume"].clip(0, 1)

        return df

    def batch_process(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        step: int = 1,
        output_type: str = "vector",
    ) -> List[dict]:
        """
        Process entire DataFrame into feature vectors/images.

        Args:
            df: Raw OHLCV DataFrame
            window_size: Window size
            step: Step between windows
            output_type: 'vector', 'image', or 'both'

        Returns:
            List of dicts with processed data and metadata
        """
        results = []
        windows = self.create_windows(df, window_size, step)

        for window_df, start_idx, end_idx in windows:
            # Normalize the window
            normalized = self.normalize(window_df)

            result = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_date": window_df.index[0] if hasattr(window_df.index[0], "strftime") else str(window_df.index[0]),
                "end_date": window_df.index[-1] if hasattr(window_df.index[-1], "strftime") else str(window_df.index[-1]),
            }

            if output_type in ("vector", "both"):
                result["vector"] = self.to_feature_vector(normalized)

            if output_type in ("image", "both"):
                result["image"] = self.to_kline_image(normalized)

            results.append(result)

        return results
