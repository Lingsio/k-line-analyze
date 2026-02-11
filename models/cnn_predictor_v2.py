"""
CNN-based stock price movement predictor v2 (k-line-net-mc).

Key changes from v1:
- 10-day window (vs 60) → each candle gets ~25px width at 256x256
- 256x256 image resolution → candlestick patterns clearly visible
- K-line formations (doji, hammer, engulfing) become distinguishable

Architecture: ResNet18 backbone + multi-class prediction head.
Input: (batch, 3, 256, 256) - RGB K-line images of 10-day candles
Output: dict with classification logits + regression per horizon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional, List


# ── Constants ─────────────────────────────────────────────────────────────
IMAGE_SIZE = 256
WINDOW_SIZE = 10

CLASS_NAMES = ["跌", "涨"]


class KLineCNNPredictorV2(nn.Module):
    """
    CNN predictor for 10-day K-line images at 256x256 resolution.

    Each candle occupies ~25px width → body shape, shadows, and
    classic candlestick patterns are clearly resolvable by convolutions.

    Binary classification:
        0: 跌 (return < 0)
        1: 涨 (return >= 0)
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        predict_horizons: list = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.predict_horizons = predict_horizons or [1, 3, 5]

        # ResNet18 backbone
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

        # Per-horizon heads
        self.cls_heads = nn.ModuleDict()
        self.reg_heads = nn.ModuleDict()
        for h in self.predict_horizons:
            self.cls_heads[f"t{h}"] = nn.Linear(256, num_classes)
            self.reg_heads[f"t{h}"] = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        shared = self.shared_head(features)

        outputs = {}
        for h in self.predict_horizons:
            key = f"t{h}"
            outputs[f"cls_{key}"] = self.cls_heads[key](shared)
            outputs[f"reg_{key}"] = self.reg_heads[key](shared)
        return outputs

    def predict(self, x: torch.Tensor) -> dict:
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
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    @staticmethod
    def return_to_label(ret: float) -> int:
        """0 = 跌, 1 = 涨"""
        return 1 if ret >= 0 else 0

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "num_classes": self.num_classes,
                    "predict_horizons": self.predict_horizons,
                    "image_size": IMAGE_SIZE,
                    "window_size": WINDOW_SIZE,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "KLineCNNPredictorV2":
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(
            num_classes=config["num_classes"],
            predict_horizons=config["predict_horizons"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
