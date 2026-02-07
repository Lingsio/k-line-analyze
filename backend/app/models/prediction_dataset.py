"""
Dataset classes for stock prediction training.

Provides unified dataset interfaces for both CNN and Transformer predictors,
supporting all-stock (universal) and single-stock training modes.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class StockPredictionDataset(Dataset):
    """
    Base dataset for stock price movement prediction.

    Generates labeled samples from OHLCV data:
    - Input: a window of OHLCV data (for feature extraction)
    - Label: future return classification + regression target

    Supports both image-based (CNN) and sequence-based (Transformer) outputs.
    """

    THRESHOLDS = [-0.03, -0.01, 0.01, 0.03]

    def __init__(
        self,
        data_list: List[Dict],
        window_size: int = 60,
        predict_horizons: list = None,
        output_mode: str = "both",
        image_size: int = 128,
        transform=None,
    ):
        """
        Args:
            data_list: List of dicts with keys:
                - 'symbol': stock symbol
                - 'market': market identifier
                - 'df': pandas DataFrame with OHLCV columns (indexed by date)
            window_size: Number of bars in input window
            predict_horizons: List of prediction horizons (e.g., [1, 5, 10, 20])
            output_mode: 'image' (CNN), 'sequence' (Transformer), or 'both'
            image_size: Image size for CNN mode
            transform: torchvision transform for images
        """
        self.window_size = window_size
        self.predict_horizons = predict_horizons or [5]
        self.max_horizon = max(self.predict_horizons)
        self.output_mode = output_mode
        self.image_size = image_size
        self.transform = transform

        # Build all valid samples
        self.samples = []
        self._build_samples(data_list)

    def _build_samples(self, data_list: List[Dict]):
        """Extract all valid (window, label) pairs from the data."""
        for data_info in data_list:
            df = data_info["df"]
            symbol = data_info.get("symbol", "unknown")
            market = data_info.get("market", "unknown")

            if len(df) < self.window_size + self.max_horizon:
                continue

            # Ensure columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required_cols):
                continue

            # Slide window with step=1 for maximum training data
            for i in range(len(df) - self.window_size - self.max_horizon + 1):
                window_df = df.iloc[i : i + self.window_size]
                future_df = df.iloc[
                    i + self.window_size : i + self.window_size + self.max_horizon
                ]

                # Calculate returns for each horizon
                base_close = window_df["close"].iloc[-1]
                if base_close <= 0:
                    continue

                labels = {}
                returns = {}
                valid = True
                for h in self.predict_horizons:
                    if h <= len(future_df):
                        future_close = future_df["close"].iloc[h - 1]
                        ret = (future_close - base_close) / base_close
                        returns[h] = ret
                        labels[h] = self._return_to_label(ret)
                    else:
                        valid = False
                        break

                if not valid:
                    continue

                self.samples.append(
                    {
                        "window_df": window_df,
                        "labels": labels,
                        "returns": returns,
                        "symbol": symbol,
                        "market": market,
                    }
                )

    def _return_to_label(self, ret: float) -> int:
        """Convert return to class label."""
        for i, t in enumerate(self.THRESHOLDS):
            if ret < t:
                return i
        return len(self.THRESHOLDS)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample.

        Returns dict with:
            'image': (3, H, W) tensor (if output_mode includes image)
            'sequence': (seq_len, n_features) tensor (if output_mode includes sequence)
            'labels': dict of {horizon: class_label}
            'returns': dict of {horizon: float_return}
        """
        sample = self.samples[idx]
        window_df = sample["window_df"]
        result = {}

        if self.output_mode in ("image", "both"):
            result["image"] = self._df_to_image(window_df)

        if self.output_mode in ("sequence", "both"):
            result["sequence"] = self._df_to_sequence(window_df)

        # Labels and returns as tensors
        label_list = []
        return_list = []
        for h in self.predict_horizons:
            label_list.append(sample["labels"][h])
            return_list.append(sample["returns"][h])

        result["labels"] = torch.tensor(label_list, dtype=torch.long)
        result["returns"] = torch.tensor(return_list, dtype=torch.float32)

        return result

    def _df_to_image(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert OHLCV DataFrame to K-line image tensor."""
        from app.services.preprocessor import KLinePreprocessor

        preprocessor = KLinePreprocessor()
        normalized = preprocessor.normalize(df)
        image = preprocessor.to_kline_image(normalized, image_size=self.image_size)

        if self.transform:
            image = self.transform(image)
        else:
            # Default: HWC -> CHW, normalize to [0, 1]
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=2)
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            image = torch.from_numpy(image)

        return image

    def _df_to_sequence(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Convert OHLCV DataFrame to feature sequence tensor.

        Features per time step (9 total):
            0: normalized close
            1: normalized open
            2: normalized high
            3: normalized low
            4: normalized volume
            5: body ratio (close - open) / (high - low)
            6: upper shadow ratio
            7: lower shadow ratio
            8: log return
        """
        close = df["close"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)

        # Normalize prices by first close (percentage-based)
        base_price = close[0] if close[0] > 0 else 1.0
        norm_close = close / base_price - 1.0
        norm_open = open_ / base_price - 1.0
        norm_high = high / base_price - 1.0
        norm_low = low / base_price - 1.0

        # Normalize volume by mean
        vol_mean = np.mean(volume) if np.mean(volume) > 0 else 1.0
        norm_volume = volume / vol_mean - 1.0

        # Derived features
        wick = high - low
        wick[wick == 0] = 1e-8
        body_ratio = (close - open_) / wick
        upper_shadow = (high - np.maximum(open_, close)) / wick
        lower_shadow = (np.minimum(open_, close) - low) / wick

        # Log returns
        log_return = np.zeros_like(close)
        log_return[1:] = np.log(close[1:] / np.clip(close[:-1], 1e-8, None))

        # Stack features: (seq_len, 9)
        features = np.stack(
            [
                norm_close,
                norm_open,
                norm_high,
                norm_low,
                norm_volume,
                body_ratio,
                upper_shadow,
                lower_shadow,
                log_return,
            ],
            axis=1,
        ).astype(np.float32)

        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.from_numpy(features)

    def get_class_distribution(self) -> dict:
        """Get class distribution for each horizon (useful for class weights)."""
        distributions = {}
        for h in self.predict_horizons:
            h_idx = self.predict_horizons.index(h)
            labels = [s["labels"][h] for s in self.samples]
            unique, counts = np.unique(labels, return_counts=True)
            distributions[f"T+{h}"] = dict(zip(unique.tolist(), counts.tolist()))
        return distributions

    def compute_class_weights(self, horizon: int = None) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        horizon = horizon or self.predict_horizons[0]
        labels = [s["labels"][horizon] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)

        total = len(labels)
        n_classes = len(self.THRESHOLDS) + 1
        weights = np.ones(n_classes)
        for cls_id, count in zip(unique, counts):
            weights[cls_id] = total / (n_classes * count)

        return torch.tensor(weights, dtype=torch.float32)


def collate_prediction_batch(batch: List[dict]) -> dict:
    """Custom collate function for prediction batches."""
    result = {}

    if "image" in batch[0]:
        result["image"] = torch.stack([b["image"] for b in batch])

    if "sequence" in batch[0]:
        result["sequence"] = torch.stack([b["sequence"] for b in batch])

    result["labels"] = torch.stack([b["labels"] for b in batch])
    result["returns"] = torch.stack([b["returns"] for b in batch])

    return result


def load_stock_data_for_training(
    symbols: List[str],
    market: str,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """
    Load real stock data for training (synchronous wrapper).

    Args:
        symbols: List of stock symbols
        market: Market identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of dicts with 'symbol', 'market', 'df' keys
    """
    import asyncio
    from app.services.data_fetcher import DataFetcher

    async def _fetch_all():
        fetcher = DataFetcher()
        data_list = []
        for symbol in symbols:
            try:
                df = await fetcher.fetch_ohlcv(
                    symbol=symbol,
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                )
                if df is not None and len(df) > 60:
                    data_list.append(
                        {"symbol": symbol, "market": market, "df": df}
                    )
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
        return data_list

    return asyncio.run(_fetch_all())
