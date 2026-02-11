"""
CNN Encoder for K-line pattern embedding.

Uses a modified ResNet18 architecture to encode K-line images into
fixed-dimensional feature vectors for similarity search.

Training uses self-supervised learning with Triplet Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


class KLineEncoder(nn.Module):
    """
    CNN encoder for K-line pattern images.

    Architecture: Modified ResNet18 with custom head for embedding extraction.

    Input: (batch, 3, 128, 128) - RGB K-line images
    Output: (batch, embedding_dim) - Normalized feature vectors
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        prediction_dim: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize the encoder.

        Args:
            embedding_dim: Dimension of output embedding vector
            prediction_dim: Dimension of prediction output (e.g., 1 for regression, 2 for binary classification)
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final embedding
        """
        super().__init__()

        # Load pretrained ResNet18
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)

        # Get the feature dimension from ResNet18 (512)
        in_features = self.backbone.fc.in_features

        # Replace the classification head with embedding layers
        self.backbone.fc = nn.Identity()

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, prediction_dim),
        )

        self.embedding_dim = embedding_dim
        self.prediction_dim = prediction_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 3, H, W)

        Returns:
            Tuple:
                - L2-normalized embedding tensor (batch, embedding_dim)
                - Prediction tensor (batch, prediction_dim)
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)

        # Generate embedding
        embedding = self.embedding_head(features)
        # L2 normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)

        # Generate prediction
        prediction = self.prediction_head(features)

        return embedding, prediction

    def encode(self, x: torch.Tensor) -> np.ndarray:
        """
        Encode images to numpy arrays (for inference).

        Args:
            x: Input tensor

        Returns:
            Numpy array of embeddings
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x)
        return embedding.cpu().numpy()

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu"):
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location=device))


class TripletLoss(nn.Module):
    """
    Triplet Loss for self-supervised learning.

    Brings similar K-line patterns closer in embedding space while
    pushing dissimilar patterns apart.

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """

    def __init__(self, margin: float = 0.3):
        """
        Initialize Triplet Loss.

        Args:
            margin: Minimum distance margin between positive and negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings (batch, embedding_dim)
            positive: Positive (similar) embeddings
            negative: Negative (dissimilar) embeddings

        Returns:
            Scalar loss value
        """
        if anchor.size(0) == 0:
            return torch.tensor(0.0).to(anchor.device)
            
        # Compute pairwise distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss for Multitask Learning.
    
    Loss = alpha * TripletLoss + beta * PredictionLoss
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, margin: float = 0.3, is_classification: bool = False):
        super().__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.alpha = alpha
        self.beta = beta
        self.is_classification = is_classification
        self.prediction_loss = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    def forward(
        self,
        anchor_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        """
        # Triplet component
        t_loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        
        # Prediction component
        p_loss = self.prediction_loss(pred, target)
        
        total_loss = self.alpha * t_loss + self.beta * p_loss
        
        return total_loss, t_loss, p_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (alternative to Triplet Loss).

    Works with pairs instead of triplets.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embedding1: First embeddings
            embedding2: Second embeddings
            label: 1 if similar, 0 if dissimilar

        Returns:
            Scalar loss value
        """
        distance = F.pairwise_distance(embedding1, embedding2, p=2)

        # Similar pairs: minimize distance
        # Dissimilar pairs: maximize distance (up to margin)
        loss = label * distance.pow(2) + (1 - label) * F.relu(
            self.margin - distance
        ).pow(2)

        return loss.mean() / 2


class KLineDataset(Dataset):
    """
    Dataset for K-line images with self-supervised augmentation.

    For each sample, generates:
    - Anchor: Original K-line image
    - Positive: Augmented version of the same K-line
    - Negative: Random different K-line

    This enables self-supervised training without manual labels.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        labels: Optional[List[float]] = None,
        metadata: Optional[List[dict]] = None,
        transform=None,
        augment_transform=None,
    ):
        """
        Initialize dataset.

        Args:
            images: List of K-line images (H, W, C) as numpy arrays
            labels: List of labels (e.g., future returns) for each image
            metadata: Optional metadata for each image
            transform: Transform to apply to all images
            augment_transform: Transform for creating positive samples
        """
        self.images = images
        self.labels = labels
        self.metadata = metadata or [{}] * len(images)
        self.transform = transform
        self.augment_transform = augment_transform or self._default_augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a triplet (anchor, positive, negative) and label.

        Args:
            idx: Index of anchor image

        Returns:
            Tuple of (anchor, positive, negative, label) tensors
        """
        # Anchor
        anchor_img = self.images[idx]
        label = self.labels[idx] if self.labels is not None else 0.0

        # Positive: augmented version of anchor
        positive_img = self.augment_transform(anchor_img.copy())

        # Negative: random different image
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = np.random.randint(0, len(self.images))
        negative_img = self.images[neg_idx]

        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        else:
            # Default: convert to tensor
            anchor_img = self._to_tensor(anchor_img)
            positive_img = self._to_tensor(positive_img)
            negative_img = self._to_tensor(negative_img)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return anchor_img, positive_img, negative_img, label_tensor

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)

    def _default_augment(self, img: np.ndarray) -> np.ndarray:
        """Default augmentation for positive samples."""
        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale K-line encoder.

    Processes K-line patterns at multiple time scales (e.g., 20, 60, 120 days)
    and fuses the features into a single embedding.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        scales: List[int] = None,
    ):
        """
        Initialize multi-scale encoder.

        Args:
            embedding_dim: Final embedding dimension
            scales: List of scale identifiers (for separate encoders)
        """
        super().__init__()

        scales = scales or [20, 60, 120]
        self.scales = scales
        self.n_scales = len(scales)

        # Individual encoders for each scale
        self.encoders = nn.ModuleList([
            KLineEncoder(embedding_dim=embedding_dim, pretrained=True)
            for _ in scales
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * self.n_scales, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with multi-scale inputs.

        Args:
            inputs: List of tensors, one per scale

        Returns:
            Fused embedding tensor
        """
        if len(inputs) != self.n_scales:
            raise ValueError(f"Expected {self.n_scales} inputs, got {len(inputs)}")

        # Encode each scale
        embeddings = []
        for encoder, x in zip(self.encoders, inputs):
            embeddings.append(encoder(x))

        # Concatenate and fuse
        concat = torch.cat(embeddings, dim=1)
        fused = self.fusion(concat)

        # L2 normalize
        fused = F.normalize(fused, p=2, dim=1)

        return fused


def create_default_transforms():
    """Create default image transforms for training and inference."""
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    inference_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, inference_transform
