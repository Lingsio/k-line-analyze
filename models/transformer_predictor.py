"""
Transformer-based stock price movement predictor.

Uses self-attention mechanism to capture temporal dependencies
in OHLCV sequences for predicting future price movements.

Supports:
- All-stock training (universal model)
- Single-stock training (specialized model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class KLineTransformerPredictor(nn.Module):
    """
    Transformer encoder for K-line sequence prediction.

    Input: (batch, seq_len, n_features) - OHLCV + derived features
    Output: (batch, num_classes) - price movement class probabilities

    Binary classification:
        0: 跌 (return < 0)
        1: 涨 (return >= 0)
    """

    CLASS_NAMES = ["跌", "涨"]

    def __init__(
        self,
        n_features: int = 9,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        max_seq_len: int = 256,
        predict_horizons: list = None,
    ):
        """
        Args:
            n_features: Number of input features per time step
                        (open, high, low, close, volume, body_ratio,
                         upper_shadow, lower_shadow, return)
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            num_classes: Number of prediction classes
            max_seq_len: Maximum sequence length
            predict_horizons: List of prediction horizons in days (e.g., [1, 5, 10, 20])
        """
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.num_classes = num_classes
        self.predict_horizons = predict_horizons or [5]

        # Input projection: features -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Prediction heads - one per horizon
        self.prediction_heads = nn.ModuleDict()
        for h in self.predict_horizons:
            self.prediction_heads[f"t{h}"] = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

        # Regression head for exact return prediction (auxiliary task)
        self.regression_heads = nn.ModuleDict()
        for h in self.predict_horizons:
            self.regression_heads[f"t{h}"] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, n_features) input sequence
            src_key_padding_mask: (batch, seq_len+1) True for padding positions

        Returns:
            dict with keys:
                'cls_{horizon}': (batch, num_classes) logits per horizon
                'reg_{horizon}': (batch, 1) predicted return per horizon
        """
        batch_size = x.size(0)

        # Project input features to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Extract [CLS] token representation
        cls_output = x[:, 0, :]  # (batch, d_model)

        # Predictions for each horizon
        outputs = {}
        for h in self.predict_horizons:
            key = f"t{h}"
            outputs[f"cls_{key}"] = self.prediction_heads[key](cls_output)
            outputs[f"reg_{key}"] = self.regression_heads[key](cls_output)

        return outputs

    def predict(self, x: torch.Tensor) -> dict:
        """
        Inference-time prediction with probabilities.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            dict per horizon: {class_probs, predicted_class, predicted_return}
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

    @staticmethod
    def return_to_label(ret: float) -> int:
        """0 = 跌, 1 = 涨"""
        return 1 if ret >= 0 else 0

    def save(self, path: str):
        """Save model with config."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "n_features": self.n_features,
                    "d_model": self.d_model,
                    "num_classes": self.num_classes,
                    "predict_horizons": self.predict_horizons,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "KLineTransformerPredictor":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(
            n_features=config["n_features"],
            d_model=config["d_model"],
            num_classes=config["num_classes"],
            predict_horizons=config["predict_horizons"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model


class TransformerPredictionLoss(nn.Module):
    """
    Combined loss for transformer predictor.

    Combines:
    - Cross-entropy for classification (main task)
    - MSE for return regression (auxiliary task)
    - Label smoothing for regularization
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_weight: Optional[torch.Tensor] = None,
        regression_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing
        )
        self.regression_loss = nn.SmoothL1Loss()
        self.regression_weight = regression_weight

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_preds: torch.Tensor,
        reg_targets: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            cls_logits: (batch, num_classes) classification logits
            cls_targets: (batch,) class labels (long)
            reg_preds: (batch, 1) predicted returns
            reg_targets: (batch,) actual returns

        Returns:
            dict with 'total', 'cls_loss', 'reg_loss'
        """
        cls_loss = self.classification_loss(cls_logits, cls_targets)
        reg_loss = self.regression_loss(reg_preds.squeeze(-1), reg_targets)
        total = cls_loss + self.regression_weight * reg_loss

        return {
            "total": total,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
        }
