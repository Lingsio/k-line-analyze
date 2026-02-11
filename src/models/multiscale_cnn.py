"""
Multi-Scale CNN for K-line Pattern Recognition

Combines multiple time scales (5, 10, 20 days) for richer feature representation.
Based on Xiu et al. (2021) finding that multi-scale inputs improve prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class MultiScaleKLineEncoder(nn.Module):
    """
    Multi-scale K-line encoder that processes 5, 10, and 20-day windows.
    
    Architecture:
    - Separate ResNet18 encoders for each scale
    - Feature fusion via attention mechanism
    - Final prediction head
    
    Input: Dict with '5d', '10d', '20d' images
    Output: Classification logits
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Separate encoders for each scale
        self.encoder_5d = self._create_encoder(input_channels, pretrained)
        self.encoder_10d = self._create_encoder(input_channels, pretrained)
        self.encoder_20d = self._create_encoder(input_channels, pretrained)
        
        # Feature dimensions from ResNet18
        self.feature_dim = 512
        
        # Scale-specific projection layers
        self.proj_5d = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_10d = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_20d = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Cross-scale attention
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        # Scale importance weighting
        self.scale_weights = nn.Sequential(
            nn.Linear(embedding_dim * 3, 3),
            nn.Softmax(dim=-1),
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        
        # Prediction head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
        
    def _create_encoder(self, input_channels, pretrained):
        """Create a ResNet18 encoder."""
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify first conv for different input channels
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove final FC layer
        backbone.fc = nn.Identity()
        return backbone
    
    def extract_features(self, encoder, x):
        """Extract features from backbone."""
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.relu(x)
        x = encoder.maxpool(x)
        
        x = encoder.layer1(x)
        x = encoder.layer2(x)
        x = encoder.layer3(x)
        x = encoder.layer4(x)
        
        x = encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x_dict):
        """
        Forward pass with multi-scale inputs.
        
        Args:
            x_dict: Dict with keys '5d', '10d', '20d' containing images
                   Each should be (batch, C, H, W)
        
        Returns:
            Classification logits (batch, num_classes)
        """
        # Extract features from each scale
        feat_5d = self.extract_features(self.encoder_5d, x_dict['5d'])
        feat_10d = self.extract_features(self.encoder_10d, x_dict['10d'])
        feat_20d = self.extract_features(self.encoder_20d, x_dict['20d'])
        
        # Project to common embedding space
        emb_5d = self.proj_5d(feat_5d)  # (batch, embedding_dim)
        emb_10d = self.proj_10d(feat_10d)
        emb_20d = self.proj_20d(feat_20d)
        
        # Stack for attention: (batch, 3, embedding_dim)
        embs = torch.stack([emb_5d, emb_10d, emb_20d], dim=1)
        
        # Self-attention across scales
        attended, _ = self.scale_attention(embs, embs, embs)
        
        # Concatenate original and attended features
        combined = torch.cat([
            emb_5d, emb_10d, emb_20d,
            attended[:, 0], attended[:, 1], attended[:, 2]
        ], dim=-1)
        
        # Scale weighting
        scale_importance = self.scale_weights(combined)
        
        # Weighted fusion
        weighted_emb = (
            scale_importance[:, 0:1] * emb_5d +
            scale_importance[:, 1:2] * emb_10d +
            scale_importance[:, 2:3] * emb_20d
        )
        
        # Final fusion
        fused = self.fusion(torch.cat([emb_5d, emb_10d, emb_20d], dim=-1))
        
        # Residual connection with weighted features
        fused = fused + weighted_emb
        
        # Classify
        logits = self.classifier(fused)
        
        return logits


class HierarchicalMultiScaleCNN(nn.Module):
    """
    Hierarchical multi-scale CNN with progressive fusion.
    
    Processes scales from fine to coarse, gradually building
    a comprehensive representation.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Encoders for each scale
        self.encoder_5d = self._create_encoder(input_channels, pretrained)
        self.encoder_10d = self._create_encoder(input_channels, pretrained)
        self.encoder_20d = self._create_encoder(input_channels, pretrained)
        
        feature_dim = 512
        hidden_dim = 256
        
        # Hierarchical fusion
        # 5d -> 10d
        self.fuse_5d_to_10d = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # (5d+10d) -> 20d
        self.fuse_to_20d = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
    def _create_encoder(self, input_channels, pretrained):
        """Create ResNet18 encoder."""
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        backbone.fc = nn.Identity()
        return backbone
    
    def extract_features(self, encoder, x):
        """Extract features."""
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.relu(x)
        x = encoder.maxpool(x)
        x = encoder.layer1(x)
        x = encoder.layer2(x)
        x = encoder.layer3(x)
        x = encoder.layer4(x)
        x = encoder.avgpool(x)
        return torch.flatten(x, 1)
    
    def forward(self, x_dict):
        """
        Hierarchical forward pass.
        
        Args:
            x_dict: Dict with '5d', '10d', '20d' images
        
        Returns:
            Classification logits
        """
        # Extract features
        feat_5d = self.extract_features(self.encoder_5d, x_dict['5d'])
        feat_10d = self.extract_features(self.encoder_10d, x_dict['10d'])
        feat_20d = self.extract_features(self.encoder_20d, x_dict['20d'])
        
        # Hierarchical fusion
        # Step 1: Fuse 5d into 10d
        combined_10d = torch.cat([feat_5d, feat_10d], dim=-1)
        fused_10d = self.fuse_5d_to_10d(combined_10d)
        
        # Step 2: Fuse (5d+10d) into 20d
        combined_20d = torch.cat([fused_10d, feat_20d], dim=-1)
        final_features = self.fuse_to_20d(combined_20d)
        
        # Classify
        logits = self.classifier(final_features)
        
        return logits


class LightweightMultiScaleCNN(nn.Module):
    """
    Lightweight multi-scale CNN with shared weights.
    
    Uses a single encoder with different input sizes,
    sharing weights across scales for efficiency.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        if input_channels != 3:
            self.shared_encoder.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.shared_encoder.fc = nn.Identity()
        
        feature_dim = 512
        
        # Adaptive pooling for different input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Prediction heads per scale
        self.head_5d = nn.Linear(feature_dim, num_classes)
        self.head_10d = nn.Linear(feature_dim, num_classes)
        self.head_20d = nn.Linear(feature_dim, num_classes)
        self.head_fused = nn.Linear(feature_dim, num_classes)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(4) / 4)
        
    def extract_features(self, x):
        """Extract features using shared encoder."""
        x = self.shared_encoder.conv1(x)
        x = self.shared_encoder.bn1(x)
        x = self.shared_encoder.relu(x)
        x = self.shared_encoder.maxpool(x)
        x = self.shared_encoder.layer1(x)
        x = self.shared_encoder.layer2(x)
        x = self.shared_encoder.layer3(x)
        x = self.shared_encoder.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
    
    def forward(self, x_dict):
        """
        Forward pass with shared encoder.
        
        Args:
            x_dict: Dict with '5d', '10d', '20d' images
        
        Returns:
            Ensemble logits
        """
        # Extract features (shared weights)
        feat_5d = self.extract_features(x_dict['5d'])
        feat_10d = self.extract_features(x_dict['10d'])
        feat_20d = self.extract_features(x_dict['20d'])
        
        # Individual predictions
        logit_5d = self.head_5d(feat_5d)
        logit_10d = self.head_10d(feat_10d)
        logit_20d = self.head_20d(feat_20d)
        
        # Fused prediction
        fused = self.scale_fusion(torch.cat([feat_5d, feat_10d, feat_20d], dim=-1))
        logit_fused = self.head_fused(fused)
        
        # Soft ensemble with learned weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_logits = (
            weights[0] * logit_5d +
            weights[1] * logit_10d +
            weights[2] * logit_20d +
            weights[3] * logit_fused
        )
        
        return ensemble_logits
