"""
Dataset classes for K-Line pattern training and evaluation.

Includes:
- KLineDataset: Basic dataset for K-line windows
- TripletDataset: Triplet sampling for contrastive learning
- EvalDataset: Dataset for evaluation with ground truth
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Callable, Union
from pathlib import Path
import pickle
import random
from PIL import Image
import io


class KLineDataset(Dataset):
    """
    Basic K-Line dataset for time series windows.

    Can return either:
    - Raw OHLCV data (for sequence models like LSTM)
    - Rendered images (for CNN models)
    """

    def __init__(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        metadata: Optional[List[Dict]] = None,
        transform: Optional[Callable] = None,
        return_images: bool = False,
        image_size: int = 128,
        renderer: Optional[Callable] = None
    ):
        """
        Initialize dataset.

        Args:
            data: Array of shape (N, seq_len, features) or list of arrays
            metadata: Optional metadata for each sample
            transform: Optional transform to apply
            return_images: If True, render K-line images
            image_size: Size of rendered images
            renderer: Custom rendering function
        """
        if isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data

        self.metadata = metadata or [{}] * len(self.data)
        self.transform = transform
        self.return_images = return_images
        self.image_size = image_size
        self.renderer = renderer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        sample = self.data[idx].copy()

        if self.return_images and self.renderer is not None:
            # Render K-line image
            sample = self.renderer(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)

        return sample, self.metadata[idx]


class TripletDataset(Dataset):
    """
    Dataset for triplet-based contrastive learning.

    Generates (anchor, positive, negative) triplets where:
    - Anchor: Original sample
    - Positive: Augmented version of anchor
    - Negative: Different sample from dataset
    """

    def __init__(
        self,
        data: np.ndarray,
        augmentation: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        negative_strategy: str = 'random',
        return_images: bool = False,
        renderer: Optional[Callable] = None,
        image_size: int = 128
    ):
        """
        Initialize triplet dataset.

        Args:
            data: Array of shape (N, seq_len, features)
            augmentation: Augmentation function for positive samples
            transform: Final transform to apply
            negative_strategy: 'random' or 'hard' (semi-hard mining)
            return_images: If True, render K-line images
            renderer: K-line rendering function
            image_size: Size of rendered images
        """
        self.data = data
        self.augmentation = augmentation or self._default_augmentation
        self.transform = transform
        self.negative_strategy = negative_strategy
        self.return_images = return_images
        self.renderer = renderer
        self.image_size = image_size

        # Pre-compute indices for negative sampling
        self.indices = list(range(len(data)))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Anchor
        anchor = self.data[idx].copy()

        # Positive (augmented anchor)
        positive = self.augmentation(anchor.copy())

        # Negative (random different sample)
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = random.choice(self.indices)
        negative = self.data[neg_idx].copy()

        # Apply rendering if needed
        if self.return_images and self.renderer is not None:
            anchor = self.renderer(anchor)
            positive = self.renderer(positive)
            negative = self.renderer(negative)

        # Apply transforms
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        # Convert to tensors
        if not isinstance(anchor, torch.Tensor):
            anchor = torch.tensor(anchor, dtype=torch.float32)
            positive = torch.tensor(positive, dtype=torch.float32)
            negative = torch.tensor(negative, dtype=torch.float32)

        return anchor, positive, negative

    def _default_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Default augmentation: add Gaussian noise."""
        noise = np.random.randn(*x.shape) * 0.01
        return x + noise


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning (SimCLR-style).

    Returns two augmented views of the same sample.
    """

    def __init__(
        self,
        data: np.ndarray,
        augmentation1: Optional[Callable] = None,
        augmentation2: Optional[Callable] = None,
        transform: Optional[Callable] = None
    ):
        self.data = data
        self.augmentation1 = augmentation1 or self._default_augmentation
        self.augmentation2 = augmentation2 or self._default_augmentation
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx].copy()

        view1 = self.augmentation1(sample.copy())
        view2 = self.augmentation2(sample.copy())

        if self.transform is not None:
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        if not isinstance(view1, torch.Tensor):
            view1 = torch.tensor(view1, dtype=torch.float32)
            view2 = torch.tensor(view2, dtype=torch.float32)

        return view1, view2

    def _default_augmentation(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*x.shape) * 0.01
        return x + noise


class EvalDataset(Dataset):
    """
    Dataset for evaluation with ground truth labels.

    Supports:
    - Pattern type labels (for classification-based eval)
    - Similarity pairs (for retrieval-based eval)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        similar_pairs: Optional[List[Tuple[int, int]]] = None,
        metadata: Optional[List[Dict]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize evaluation dataset.

        Args:
            data: Array of shape (N, seq_len, features)
            labels: Optional label array for classification
            similar_pairs: Optional list of (idx1, idx2) similar pairs
            metadata: Optional metadata for each sample
            transform: Optional transform
        """
        self.data = data
        self.labels = labels
        self.similar_pairs = similar_pairs
        self.metadata = metadata or [{}] * len(data)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx].copy()

        if self.transform is not None:
            sample = self.transform(sample)

        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)

        result = {
            'data': sample,
            'index': idx,
            'metadata': self.metadata[idx]
        }

        if self.labels is not None:
            result['label'] = self.labels[idx]

        return result

    def get_similar_items(self, idx: int) -> List[int]:
        """Get indices of items similar to the given index."""
        if self.similar_pairs is None:
            return []

        similar = []
        for i, j in self.similar_pairs:
            if i == idx:
                similar.append(j)
            elif j == idx:
                similar.append(i)

        return similar

    def get_items_with_label(self, label: int) -> List[int]:
        """Get indices of items with the given label."""
        if self.labels is None:
            return []
        return np.where(self.labels == label)[0].tolist()


class DataAugmentation:
    """Collection of data augmentation methods for K-line data."""

    @staticmethod
    def add_noise(x: np.ndarray, std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.randn(*x.shape) * std
        return x + noise

    @staticmethod
    def scale(x: np.ndarray, factor_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Random scaling."""
        factor = np.random.uniform(*factor_range)
        return x * factor

    @staticmethod
    def shift(x: np.ndarray, shift_range: Tuple[float, float] = (-0.05, 0.05)) -> np.ndarray:
        """Random vertical shift."""
        shift = np.random.uniform(*shift_range)
        return x + shift

    @staticmethod
    def time_warp(x: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Random time warping (stretch/compress time axis).

        Args:
            x: Input of shape (seq_len, features)
            sigma: Warping intensity
        """
        seq_len = len(x)
        orig_steps = np.arange(seq_len)

        # Generate random warping path
        random_warps = np.random.randn(seq_len) * sigma
        warp_steps = np.clip(orig_steps + random_warps, 0, seq_len - 1)
        warp_steps = np.sort(warp_steps)  # Ensure monotonicity

        # Interpolate
        result = np.zeros_like(x)
        for i in range(x.shape[1] if x.ndim > 1 else 1):
            if x.ndim > 1:
                result[:, i] = np.interp(orig_steps, warp_steps, x[:, i])
            else:
                result = np.interp(orig_steps, warp_steps, x)

        return result

    @staticmethod
    def window_crop(x: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
        """
        Random window cropping (then resize to original length).

        Args:
            x: Input of shape (seq_len, features)
            crop_ratio: Ratio of original length to keep
        """
        seq_len = len(x)
        crop_len = int(seq_len * crop_ratio)

        if crop_len >= seq_len:
            return x.copy()

        start = np.random.randint(0, seq_len - crop_len)
        cropped = x[start:start + crop_len]

        # Resize back to original length
        orig_steps = np.linspace(0, crop_len - 1, seq_len)
        crop_steps = np.arange(crop_len)

        result = np.zeros_like(x)
        for i in range(x.shape[1] if x.ndim > 1 else 1):
            if x.ndim > 1:
                result[:, i] = np.interp(orig_steps, crop_steps, cropped[:, i])
            else:
                result = np.interp(orig_steps, crop_steps, cropped)

        return result

    @staticmethod
    def combine(*augmentations) -> Callable:
        """Combine multiple augmentations into one function."""
        def combined(x):
            for aug in augmentations:
                x = aug(x)
            return x
        return combined


def create_data_splits(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets.

    Args:
        data: Input data array
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        shuffle: Whether to shuffle before splitting
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(data)
    indices = np.arange(n)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return data[train_indices], data[val_indices], data[test_indices]


def create_triplet_dataloader(
    data: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    augmentation: Optional[Callable] = None,
    **kwargs
) -> DataLoader:
    """Create DataLoader for triplet training."""
    dataset = TripletDataset(data, augmentation=augmentation, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == '__main__':
    # Test datasets
    print("Testing dataset classes...")

    # Create dummy data
    np.random.seed(42)
    dummy_data = np.random.randn(100, 60, 5)  # 100 samples, 60 timesteps, 5 features

    # Test KLineDataset
    print("\n1. Testing KLineDataset...")
    dataset = KLineDataset(dummy_data)
    sample, meta = dataset[0]
    print(f"   Sample shape: {sample.shape}")

    # Test TripletDataset
    print("\n2. Testing TripletDataset...")
    triplet_dataset = TripletDataset(dummy_data)
    anchor, positive, negative = triplet_dataset[0]
    print(f"   Anchor shape: {anchor.shape}")
    print(f"   Positive shape: {positive.shape}")
    print(f"   Negative shape: {negative.shape}")

    # Test DataLoader
    print("\n3. Testing DataLoader...")
    loader = create_triplet_dataloader(dummy_data, batch_size=8)
    for batch in loader:
        a, p, n = batch
        print(f"   Batch shapes: {a.shape}, {p.shape}, {n.shape}")
        break

    # Test augmentations
    print("\n4. Testing augmentations...")
    sample = dummy_data[0]
    aug = DataAugmentation()
    print(f"   Original shape: {sample.shape}")
    print(f"   Noisy shape: {aug.add_noise(sample).shape}")
    print(f"   Scaled shape: {aug.scale(sample).shape}")
    print(f"   Time-warped shape: {aug.time_warp(sample).shape}")

    # Test data splits
    print("\n5. Testing data splits...")
    train, val, test = create_data_splits(dummy_data)
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print("\nAll tests passed!")
