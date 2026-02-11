"""
Lightweight CNN for sparse grayscale OHLC bar chart images.

Designed for sector-based grouped stock prediction with quantile filtering.

Architecture for 20-day window (input: 1 x 64 x 60):
  Layer1: Conv2d(1->64,  5x3, stride=3x1, dil=2x1, pad=2x1) -> BN -> LeakyReLU -> MaxPool(2x1)
  Layer2: Conv2d(64->128, 5x3, stride=1x1, dil=1x1, pad=2x1) -> BN -> LeakyReLU -> MaxPool(2x1)
  Layer3: Conv2d(128->256, 5x3, stride=1x1, dil=1x1, pad=2x1) -> BN -> LeakyReLU -> MaxPool(2x1)
  Flatten -> Dropout(0.5) -> Linear(flatten_size, num_classes)

~709K parameters (20-day). Xavier uniform init.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


# Image sizes per window size: {window_days: (H, W)}
SPARSE_IMAGE_SIZES = {
    5:  (32, 15),
    20: (64, 60),
    60: (96, 180),
}

# Layer counts per window size
SPARSE_LAYER_COUNTS = {5: 2, 20: 3, 60: 4}

# Architecture settings per window size:
# (filter_sizes, strides, dilations, max_pools)
SPARSE_ARCH_SETTINGS = {
    5: {
        'filters':   [(5, 3)] * 2,
        'strides':   [(1, 1)] * 2,
        'dilations': [(1, 1)] * 2,
        'maxpools':  [(2, 1)] * 2,
    },
    20: {
        'filters':   [(5, 3)] * 3,
        'strides':   [(3, 1)] + [(1, 1)] * 2,
        'dilations': [(2, 1)] + [(1, 1)] * 2,
        'maxpools':  [(2, 1)] * 3,
    },
    60: {
        'filters':   [(5, 3)] * 4,
        'strides':   [(3, 1)] + [(1, 1)] * 3,
        'dilations': [(3, 1)] + [(1, 1)] * 3,
        'maxpools':  [(2, 1)] * 4,
    },
}


class SparseCNN(nn.Module):
    """
    Lightweight CNN for sparse grayscale OHLC images.

    Binary classification: 0 = down, 1 = up.
    """

    def __init__(self, num_classes=2, window_size=20, inplanes=64, dropout=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.window_size = window_size
        self.image_size = SPARSE_IMAGE_SIZES.get(window_size, (64, window_size * 3))

        num_layers = SPARSE_LAYER_COUNTS.get(window_size, 3)
        arch = SPARSE_ARCH_SETTINGS.get(window_size, SPARSE_ARCH_SETTINGS[20])

        # Build convolutional layers
        layers = []
        in_channels = 1  # Grayscale input
        out_channels = inplanes

        for i in range(num_layers):
            filt = arch['filters'][i]
            stride = arch['strides'][i]
            dil = arch['dilations'][i]
            mp = arch['maxpools'][i]

            # Padding = int(filter_size / 2) per dimension
            pad = (filt[0] // 2, filt[1] // 2)

            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=filt, stride=stride,
                padding=pad, dilation=dil,
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=mp, ceil_mode=True))

            in_channels = out_channels
            out_channels = in_channels * 2  # Double channels each layer

        self.features = nn.Sequential(*layers)

        # Compute FC input size via dummy forward pass
        self._flatten_size = self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._flatten_size, num_classes),
        )

        # Xavier uniform initialization
        self._init_weights()

    def _get_flatten_size(self):
        H, W = self.image_size
        dummy = torch.zeros(1, 1, H, W)
        with torch.no_grad():
            out = self.features(dummy)
        return out.view(1, -1).size(1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, H, W) grayscale image tensor

        Returns:
            logits: (batch, num_classes)
        """
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'window_size': self.window_size,
                'image_size': self.image_size,
            },
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(
            num_classes=config['num_classes'],
            window_size=config['window_size'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model
