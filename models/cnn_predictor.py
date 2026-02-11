"""
CNN-based stock price movement predictor (k-line-net-mc).

Reuses the ResNet18 backbone from KLineEncoder but replaces the
embedding head with a multi-class prediction head.

Supports:
- All-stock training (universal model)
- Single-stock training (specialized model per ticker)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional, List


class KLineCNNPredictor(nn.Module):
    """
    CNN predictor for K-line pattern images (k-line-net-mc).

    Architecture: ResNet18 backbone + multi-class prediction head.

    Input: (batch, 3, 128, 128) - RGB K-line images
    Output: dict with classification logits and regression values per horizon

    Classes (5-class):
        0: strong_bearish  (< -3%)
        1: bearish          (-3% ~ -1%)
        2: neutral          (-1% ~ +1%)
        3: bullish          (+1% ~ +3%)
        4: strong_bullish   (> +3%)
    """

    THRESHOLDS = [-0.03, -0.01, 0.01, 0.03]
    CLASS_NAMES = ["strong_bearish", "bearish", "neutral", "bullish", "strong_bullish"]

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        predict_horizons: list = None,
        freeze_backbone_epochs: int = 0,
    ):
        """
        Args:
            num_classes: Number of prediction classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate
            predict_horizons: Prediction horizons in days (e.g., [1, 5, 10, 20])
            freeze_backbone_epochs: Number of epochs to freeze backbone (for fine-tuning)
        """
        super().__init__()

        self.num_classes = num_classes
        self.predict_horizons = predict_horizons or [5]
        self.freeze_backbone_epochs = freeze_backbone_epochs

        # Load pretrained ResNet18
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)

        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Shared feature layer
        self.shared_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # Classification heads - one per horizon
        self.cls_heads = nn.ModuleDict()
        for h in self.predict_horizons:
            self.cls_heads[f"t{h}"] = nn.Linear(256, num_classes)

        # Regression heads - one per horizon
        self.reg_heads = nn.ModuleDict()
        for h in self.predict_horizons:
            self.reg_heads[f"t{h}"] = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            x: (batch, 3, H, W) input K-line image

        Returns:
            dict with 'cls_{horizon}' and 'reg_{horizon}' keys
        """
        features = self.backbone(x)  # (batch, 512)
        shared = self.shared_head(features)  # (batch, 256)

        outputs = {}
        for h in self.predict_horizons:
            key = f"t{h}"
            outputs[f"cls_{key}"] = self.cls_heads[key](shared)
            outputs[f"reg_{key}"] = self.reg_heads[key](shared)

        return outputs

    def predict(self, x: torch.Tensor) -> dict:
        """
        Inference-time prediction with probabilities.

        Args:
            x: (batch, 3, H, W)

        Returns:
            dict per horizon with class probs, predicted class, predicted return
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

        results = {}
        for h in self.predict_horizons:
            key = f"t{h}"
            logits = outputs[f"cls_{key}"]
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1)
            pred_return = outputs[f"reg_{key}"].squeeze(-1)

            results[f"T+{h}"] = {
                "class_probs": probs.cpu().numpy(),
                "predicted_class": pred_class.cpu().numpy(),
                "predicted_class_name": [
                    self.CLASS_NAMES[c] for c in pred_class.cpu().numpy()
                ],
                "predicted_return": pred_return.cpu().numpy(),
            }

        return results

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @staticmethod
    def return_to_label(ret: float, thresholds=None) -> int:
        """Convert a return value to class label."""
        thresholds = thresholds or [-0.03, -0.01, 0.01, 0.03]
        for i, t in enumerate(thresholds):
            if ret < t:
                return i
        return len(thresholds)

    def save(self, path: str):
        """Save model with config."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "num_classes": self.num_classes,
                    "predict_horizons": self.predict_horizons,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "KLineCNNPredictor":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(
            num_classes=config["num_classes"],
            predict_horizons=config["predict_horizons"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
