"""
Dataset v2 for stock prediction training.

Key changes from v1:
- Default 10-day window, 256x256 images
- Time-based train/val split (prevents data leakage)
- Horizons default to [1, 3, 5] (matching short-term 10-day input)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


# ── Constants ─────────────────────────────────────────────────────────────
DEFAULT_WINDOW_SIZE = 10
DEFAULT_IMAGE_SIZE = 256
DEFAULT_HORIZONS = [1, 3, 5]
NUM_CLASSES = 2
CLASS_NAMES = ["跌", "涨"]


class StockPredictionDatasetV2(Dataset):
    """
    Dataset for stock price movement prediction (v2).

    10-day window → 256x256 K-line image or 10-step sequence.
    Binary classification: 跌 (0) / 涨 (1).
    Time-ordered samples per stock for proper train/val splitting.
    """

    def __init__(
        self,
        data_list: List[Dict],
        window_size: int = DEFAULT_WINDOW_SIZE,
        predict_horizons: list = None,
        output_mode: str = "both",
        image_size: int = DEFAULT_IMAGE_SIZE,
        transform=None,
    ):
        """
        Args:
            data_list: List of dicts with 'symbol', 'market', 'df' keys
            window_size: Number of candle bars per sample (default: 10)
            predict_horizons: Prediction forward days (default: [1, 3, 5])
            output_mode: 'image' (CNN), 'sequence' (Transformer), or 'both'
            image_size: Image resolution (default: 256)
            transform: torchvision transform for images
        """
        self.window_size = window_size
        self.predict_horizons = predict_horizons or DEFAULT_HORIZONS
        self.max_horizon = max(self.predict_horizons)
        self.output_mode = output_mode
        self.image_size = image_size
        self.transform = transform

        self.samples = []
        self._build_samples(data_list)

    def _build_samples(self, data_list: List[Dict]):
        """Extract all valid (window, label) pairs, ordered by time per stock."""
        for data_info in data_list:
            df = data_info["df"]
            symbol = data_info.get("symbol", "unknown")
            market = data_info.get("market", "unknown")

            if len(df) < self.window_size + self.max_horizon:
                continue

            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required_cols):
                continue

            # Slide window, step=1 — samples are chronological per stock
            for i in range(len(df) - self.window_size - self.max_horizon + 1):
                window_df = df.iloc[i : i + self.window_size]
                future_df = df.iloc[
                    i + self.window_size : i + self.window_size + self.max_horizon
                ]

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
        """0 = 跌, 1 = 涨"""
        return 1 if ret >= 0 else 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        window_df = sample["window_df"]
        result = {}

        if self.output_mode in ("image", "both"):
            result["image"] = self._df_to_image(window_df)

        if self.output_mode in ("sequence", "both"):
            result["sequence"] = self._df_to_sequence(window_df)

        label_list = []
        return_list = []
        for h in self.predict_horizons:
            label_list.append(sample["labels"][h])
            return_list.append(sample["returns"][h])

        result["labels"] = torch.tensor(label_list, dtype=torch.long)
        result["returns"] = torch.tensor(return_list, dtype=torch.float32)
        return result

    def _df_to_image(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert OHLCV to 256x256 K-line image."""
        from core.preprocessor import KLinePreprocessor

        preprocessor = KLinePreprocessor()
        normalized = preprocessor.normalize(df)
        image = preprocessor.to_kline_image(normalized, image_size=self.image_size)

        if self.transform:
            image = self.transform(image)
        else:
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=2)
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            image = torch.from_numpy(image)
        return image

    def _df_to_sequence(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Convert OHLCV to feature sequence (9 features per step).

        Features: norm_close, norm_open, norm_high, norm_low,
                  norm_volume, body_ratio, upper_shadow, lower_shadow, log_return
        """
        close = df["close"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)

        base_price = close[0] if close[0] > 0 else 1.0
        norm_close = close / base_price - 1.0
        norm_open = open_ / base_price - 1.0
        norm_high = high / base_price - 1.0
        norm_low = low / base_price - 1.0

        vol_mean = np.mean(volume) if np.mean(volume) > 0 else 1.0
        norm_volume = volume / vol_mean - 1.0

        wick = high - low
        wick[wick == 0] = 1e-8
        body_ratio = (close - open_) / wick
        upper_shadow = (high - np.maximum(open_, close)) / wick
        lower_shadow = (np.minimum(open_, close) - low) / wick

        log_return = np.zeros_like(close)
        log_return[1:] = np.log(close[1:] / np.clip(close[:-1], 1e-8, None))

        features = np.stack(
            [norm_close, norm_open, norm_high, norm_low, norm_volume,
             body_ratio, upper_shadow, lower_shadow, log_return],
            axis=1,
        ).astype(np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.from_numpy(features)

    def get_class_distribution(self) -> dict:
        distributions = {}
        for h in self.predict_horizons:
            labels = [s["labels"][h] for s in self.samples]
            unique, counts = np.unique(labels, return_counts=True)
            distributions[f"T+{h}"] = dict(zip(unique.tolist(), counts.tolist()))
        return distributions

    def compute_class_weights(self, horizon: int = None) -> torch.Tensor:
        horizon = horizon or self.predict_horizons[0]
        labels = [s["labels"][horizon] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)

        total = len(labels)
        weights = np.ones(NUM_CLASSES)
        for cls_id, count in zip(unique, counts):
            weights[cls_id] = total / (NUM_CLASSES * count)
        return torch.tensor(weights, dtype=torch.float32)


# ── Time-based split ──────────────────────────────────────────────────────

def time_based_split(
    dataset: StockPredictionDatasetV2,
    val_ratio: float = 0.15,
) -> Tuple[Subset, Subset]:
    """
    Split dataset chronologically per stock to prevent data leakage.

    For each stock, the last val_ratio% of samples become validation.
    No overlapping time periods between train and val.

    Returns:
        (train_subset, val_subset) as torch Subsets
    """
    symbol_indices = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        symbol_indices[sample["symbol"]].append(i)

    train_indices = []
    val_indices = []
    for symbol, indices in symbol_indices.items():
        split_point = int(len(indices) * (1 - val_ratio))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ── Collate & data loading ────────────────────────────────────────────────

def collate_prediction_batch(batch: List[dict]) -> dict:
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
    """Load real stock data (synchronous wrapper)."""
    import asyncio
    from core.data_fetcher import DataFetcher

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
                if df is not None and len(df) > 30:
                    data_list.append(
                        {"symbol": symbol, "market": market, "df": df}
                    )
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
        return data_list

    return asyncio.run(_fetch_all())
