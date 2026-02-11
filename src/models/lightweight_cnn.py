"""
Lightweight CNN for RTX 4060 (8GB VRAM)
=======================================

Optimized for small GPU memory while maintaining good performance.
Expected memory usage: ~4-6GB with batch_size=32

Architecture:
- 4 Conv blocks with batch norm
- Global Average Pooling (reduces parameters)
- Small FC layer
- Total params: ~500K (vs ResNet18's 11M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for K-line pattern recognition.
    
    Args:
        num_classes: Number of output classes (2 for binary, 3 for up/neutral/down)
        input_channels: Number of input channels (3 for RGB, 1 for grayscale)
        dropout: Dropout rate
    """
    
    def __init__(self, num_classes=2, input_channels=3, dropout=0.3):
        super(LightweightCNN, self).__init__()
        
        # Block 1: 128x128 -> 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5)
        )
        
        # Block 2: 64x64 -> 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5)
        )
        
        # Block 3: 32x32 -> 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )
        
        # Block 4: 16x16 -> 8x8
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )
        
        # Global Average Pooling: 8x8x256 -> 1x1x256
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.embedding = None
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Store embedding (before classifier)
        self.embedding = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        """Extract embedding features."""
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.global_pool(x)
            return x.view(x.size(0), -1)
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UltraLightCNN(nn.Module):
    """
    Even smaller CNN for very limited GPU memory.
    Expected memory: ~2-3GB with batch_size=32
    Total params: ~100K
    """
    
    def __init__(self, num_classes=2, input_channels=3, dropout=0.3):
        super(UltraLightCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.3),
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),
            
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.embedding = None
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        self.embedding = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = self.global_pool(x)
            return x.view(x.size(0), -1)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# Factory function
def build_lightweight_cnn(variant='light', **kwargs):
    """
    Build lightweight CNN.
    
    Args:
        variant: 'light' (default) or 'ultra'
        **kwargs: Passed to model constructor
    
    Returns:
        Model instance
    """
    if variant == 'ultra':
        model = UltraLightCNN(**kwargs)
    else:
        model = LightweightCNN(**kwargs)
    
    print(f"Built {variant} CNN: {model.get_num_params():,} parameters")
    return model


if __name__ == '__main__':
    # Test model sizes
    print("=" * 60)
    print("Lightweight CNN Model Summary")
    print("=" * 60)
    
    # Light version
    model_light = build_lightweight_cnn('light', num_classes=3, input_channels=3)
    x = torch.randn(2, 3, 128, 128)
    y = model_light(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Embedding dim: {model_light.embedding.shape[1]}")
    
    print()
    
    # Ultra version
    model_ultra = build_lightweight_cnn('ultra', num_classes=3, input_channels=3)
    y = model_ultra(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Embedding dim: {model_ultra.embedding.shape[1]}")
