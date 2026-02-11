"""
High-Resolution Financial CNN optimized for H20 (96GB VRAM).

Key features:
1. High resolution support: 256x256, 512x512
2. Multi-scale feature extraction (similar to Xiu but deeper)
3. Attention mechanisms for time-series
4. Squeeze-and-Excitation blocks for channel attention
5. Dilated convolutions for larger receptive field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBNAct(nn.Module):
    """Conv + BatchNorm + Activation + optional SE."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, 
                 dilation=1, use_se=True, activation='leaky_relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_ch)
        
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_se:
            x = self.se(x)
        x = self.act(x)
        return x


class FinanceBlock(nn.Module):
    """
    Financial data specific block.
    Captures both temporal (horizontal) and price (vertical) patterns.
    """
    def __init__(self, in_ch, out_ch, stride=1, use_dilation=False):
        super().__init__()
        
        # Temporal conv (wider kernel horizontally for time patterns)
        self.conv_temporal = ConvBNAct(in_ch, out_ch // 2, 
                                       kernel_size=(3, 7), 
                                       stride=stride,
                                       padding=(1, 3))
        
        # Price conv (taller kernel vertically for price levels)
        self.conv_price = ConvBNAct(in_ch, out_ch // 2, 
                                    kernel_size=(7, 3), 
                                    stride=stride,
                                    padding=(3, 1))
        
        # Dilated conv for larger receptive field (optional)
        if use_dilation:
            self.conv_dilated = ConvBNAct(out_ch, out_ch // 4, 
                                          kernel_size=3, 
                                          padding=2, 
                                          dilation=2)
            self.out_ch = out_ch + out_ch // 4
        else:
            self.conv_dilated = None
            self.out_ch = out_ch
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(self.out_ch)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # Multi-scale features
        t = self.conv_temporal(x)
        p = self.conv_price(x)
        
        out = torch.cat([t, p], dim=1)
        
        if self.conv_dilated is not None:
            d = self.conv_dilated(out)
            out = torch.cat([out, d], dim=1)
        
        out = out + identity
        return F.leaky_relu(out, 0.1, inplace=True)


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important time/price regions."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class FinanceCNN_H20(nn.Module):
    """
    High-Resolution Financial CNN for H20 (96GB VRAM).
    
    Supports resolutions: 256x256, 512x512
    Window sizes: 20, 60 days
    """
    
    # Recommended image sizes for different resolutions
    IMAGE_CONFIGS = {
        'standard': {  # Xiu style
            20: (64, 60),
            60: (96, 180),
        },
        'high': {  # 2x resolution
            20: (128, 120),
            60: (192, 360),
        },
        'ultra': {  # 4x resolution
            20: (256, 240),
            60: (384, 720),
        },
        'h20_max': {  # Max for H20
            20: (512, 480),
            60: (512, 960),
        }
    }
    
    def __init__(
        self,
        window_size=20,
        num_classes=2,
        resolution='high',  # 'standard', 'high', 'ultra', 'h20_max'
        base_channels=128,  # Xiu uses 64, we can use 128 or 256 on H20
        num_blocks=4,       # Number of FinanceBlocks
        dropout=0.3,
        use_attention=True,
    ):
        super().__init__()
        
        self.window_size = window_size
        self.resolution = resolution
        self.img_size = self.IMAGE_CONFIGS[resolution][window_size]
        self.use_attention = use_attention
        
        # Initial conv (large kernel for initial feature extraction)
        self.stem = nn.Sequential(
            ConvBNAct(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Finance blocks with increasing channels
        self.blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** (i // 2))  # Double every 2 blocks
            stride = 2 if i % 2 == 1 else 1  # Downsample every other block
            use_dilation = (i >= num_blocks - 2)  # Use dilation in last 2 blocks
            
            self.blocks.append(FinanceBlock(in_ch, out_ch, stride, use_dilation))
            in_ch = self.blocks[-1].out_ch
        
        # Spatial attention
        if use_attention:
            self.spatial_att = SpatialAttention()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(in_ch // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        if self.use_attention:
            x = self.spatial_att(x)
        
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self):
        """Estimate FLOPs (rough calculation)."""
        h, w = self.img_size
        # Simplified FLOPs estimation
        return h * w * self.count_parameters() / 1e9  # GFLOPs


class FinanceCNN_Large(nn.Module):
    """
    Even larger model for H20 with transformer-style attention.
    Suitable for 512x512 resolution.
    """
    
    def __init__(
        self,
        window_size=20,
        num_classes=2,
        img_size=(512, 480),
        embed_dim=256,      # Embedding dimension
        num_heads=8,        # Attention heads
        num_layers=6,       # Number of transformer-style blocks
        dropout=0.2,
    ):
        super().__init__()
        
        self.img_size = img_size
        
        # CNN backbone (similar to ResNet but optimized for finance)
        self.backbone = nn.Sequential(
            # Stage 1
            ConvBNAct(1, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Stage 2
            FinanceBlock(64, 128, stride=1),
            FinanceBlock(128, 128, stride=2),
            
            # Stage 3
            FinanceBlock(128, 256, stride=1),
            FinanceBlock(256, 256, stride=2),
            FinanceBlock(256, 256, stride=1),
            
            # Stage 4
            FinanceBlock(256, 512, stride=2),
            FinanceBlock(512, 512, stride=1),
            FinanceBlock(512, 512, stride=1),
            FinanceBlock(512, embed_dim, stride=2),
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, embed_dim, 16, 16) * 0.02)
        
        # Transformer-style blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.backbone(x)  # (B, embed_dim, H', W')
        
        # Add positional encoding
        if x.size(2) >= 16 and x.size(3) >= 16:
            x = x + self.pos_encoding[:, :, :x.size(2), :x.size(3)]
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    """Simple transformer block for 2D features."""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for attention: (B, H*W, C)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        
        # Self-attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return x


def create_model(model_type='finance_h20', **kwargs):
    """Factory function to create models."""
    if model_type == 'finance_h20':
        return FinanceCNN_H20(**kwargs)
    elif model_type == 'finance_large':
        return FinanceCNN_Large(**kwargs)
    elif model_type == 'xiu':
        from .xiu_cnn import XiuCNN
        return XiuCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Memory estimates for H20 (96GB)
MEMORY_ESTIMATES = {
    'standard': {'params': 0.7e6, 'vram_per_batch': 2, 'max_batch': 256},   # 2GB per batch of 256
    'high': {'params': 5e6, 'vram_per_batch': 4, 'max_batch': 128},         # 4GB per batch of 128
    'ultra': {'params': 20e6, 'vram_per_batch': 8, 'max_batch': 64},        # 8GB per batch of 64
    'h20_max': {'params': 50e6, 'vram_per_batch': 16, 'max_batch': 32},     # 16GB per batch of 32
}
