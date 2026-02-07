"""
Training script for Transformer-based stock predictor.

Uses self-attention to capture temporal patterns in OHLCV sequences.

Supports two training modes:
1. All-stock (universal): Train on multiple stocks across markets
2. Single-stock: Train a specialized model for one specific ticker

Usage:
    # All-stock training (universal model)
    python scripts/train_transformer_predictor.py --mode all --markets us tw cn

    # Single-stock training
    python scripts/train_transformer_predictor.py --mode single --symbol AAPL --market us

    # H20 GPU optimized (larger batch/model)
    python scripts/train_transformer_predictor.py --mode all --markets us tw cn hk \
        --batch-size 128 --d-model 256 --nhead 8 --layers 6 --epochs 100
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

from app.models.transformer_predictor import (
    KLineTransformerPredictor,
    TransformerPredictionLoss,
)
from app.models.prediction_dataset import (
    StockPredictionDataset,
    collate_prediction_batch,
    load_stock_data_for_training,
)
from app.config import settings


# Default stock lists per market
STOCK_LISTS = {
    "us": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
        "NFLX", "INTC", "JPM", "BAC", "V", "MA", "JNJ", "PFE", "UNH",
        "PG", "KO", "WMT", "XOM", "CVX", "BA", "DIS", "CRM", "CSCO",
    ],
    "tw": [
        "2330", "2317", "2454", "2412", "2308", "2303", "2881", "2882",
        "1301", "1303", "2002", "1216", "2891", "2886", "2884",
    ],
    "cn": [
        "600519", "000858", "601318", "600036", "601166", "600276",
        "600887", "000333", "002594", "300750", "000001", "600000",
    ],
    "hk": [
        "0700", "9988", "0005", "0939", "1299", "0941", "3690", "1810",
    ],
}


def train_epoch(model, dataloader, criterion, optimizer, device, horizons, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = {h: 0 for h in horizons}
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        sequences = batch["sequence"].to(device)
        labels = batch["labels"].to(device)
        returns = batch["returns"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.amp.autocast("cuda"):
                outputs = model(sequences)
                loss_total = torch.tensor(0.0, device=device)
                batch_cls = 0
                batch_reg = 0

                for i, h in enumerate(horizons):
                    key = f"t{h}"
                    loss_dict = criterion(
                        outputs[f"cls_{key}"], labels[:, i],
                        outputs[f"reg_{key}"], returns[:, i],
                    )
                    loss_total = loss_total + loss_dict["total"]
                    batch_cls += loss_dict["cls_loss"].item()
                    batch_reg += loss_dict["reg_loss"].item()

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(sequences)
            loss_total = torch.tensor(0.0, device=device)
            batch_cls = 0
            batch_reg = 0

            for i, h in enumerate(horizons):
                key = f"t{h}"
                loss_dict = criterion(
                    outputs[f"cls_{key}"], labels[:, i],
                    outputs[f"reg_{key}"], returns[:, i],
                )
                loss_total = loss_total + loss_dict["total"]
                batch_cls += loss_dict["cls_loss"].item()
                batch_reg += loss_dict["reg_loss"].item()

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Accuracy tracking
        with torch.no_grad():
            for i, h in enumerate(horizons):
                key = f"t{h}"
                preds = outputs[f"cls_{key}"].argmax(dim=-1)
                correct[h] += (preds == labels[:, i]).sum().item()

        total_loss += loss_total.item()
        total_cls_loss += batch_cls
        total_reg_loss += batch_reg
        total += labels.size(0)

    n_batches = len(dataloader)
    accuracies = {h: correct[h] / total * 100 for h in horizons}
    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "reg_loss": total_reg_loss / n_batches,
        "accuracies": accuracies,
    }


def validate(model, dataloader, criterion, device, horizons):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = {h: 0 for h in horizons}
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            labels = batch["labels"].to(device)
            returns = batch["returns"].to(device)

            outputs = model(sequences)

            loss_total = torch.tensor(0.0, device=device)
            for i, h in enumerate(horizons):
                key = f"t{h}"
                loss_dict = criterion(
                    outputs[f"cls_{key}"], labels[:, i],
                    outputs[f"reg_{key}"], returns[:, i],
                )
                loss_total = loss_total + loss_dict["total"]
                preds = outputs[f"cls_{key}"].argmax(dim=-1)
                correct[h] += (preds == labels[:, i]).sum().item()

            total_loss += loss_total.item()
            total += labels.size(0)

    n_batches = len(dataloader)
    accuracies = {h: correct[h] / total * 100 for h in horizons}
    return {"loss": total_loss / n_batches, "accuracies": accuracies}


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer stock predictor (self-attention)"
    )
    # Training mode
    parser.add_argument("--mode", type=str, choices=["all", "single"], default="all",
                        help="'all' for universal model, 'single' for per-stock model")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Stock symbol (required for single mode)")
    parser.add_argument("--market", type=str, default="us")
    parser.add_argument("--markets", nargs="+", default=["us"])

    # Model architecture
    parser.add_argument("--d-model", type=int, default=128,
                        help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer encoder layers")
    parser.add_argument("--dim-ff", type=int, default=512,
                        help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training params
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (
        datetime.now() - timedelta(days=365 * 8)
    ).strftime("%Y-%m-%d")

    print("=" * 70)
    print("K-Line Transformer Predictor Training (Self-Attention)")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {device}")
    print(f"Architecture: d_model={args.d_model}, heads={args.nhead}, "
          f"layers={args.layers}, ff={args.dim_ff}")
    print(f"Horizons: {args.horizons}")
    print(f"Window size: {args.window_size}")
    print(f"Date range: {start_date} ~ {end_date}")
    print(f"Mixed Precision (AMP): {args.amp}")

    # ── Load data ─────────────────────────────────────────────────────────
    if args.mode == "single":
        if not args.symbol:
            print("ERROR: --symbol is required for single mode")
            return
        print(f"Symbol: {args.symbol} ({args.market.upper()})")
        data_list = load_stock_data_for_training(
            symbols=[args.symbol], market=args.market,
            start_date=start_date, end_date=end_date,
        )
        model_name = f"transformer_predictor_{args.market}_{args.symbol}"
    else:
        all_data = []
        for market in args.markets:
            symbols = STOCK_LISTS.get(market, [])
            print(f"Loading {market.upper()} ({len(symbols)} symbols)...")
            data = load_stock_data_for_training(
                symbols=symbols, market=market,
                start_date=start_date, end_date=end_date,
            )
            all_data.extend(data)
            print(f"  Loaded {len(data)} stocks")
        data_list = all_data
        model_name = "transformer_predictor_universal"

    if not data_list:
        print("ERROR: No data loaded. Exiting.")
        return

    print(f"Total stocks loaded: {len(data_list)}")

    # ── Create dataset (sequence mode for Transformer) ────────────────────
    dataset = StockPredictionDataset(
        data_list=data_list,
        window_size=args.window_size,
        predict_horizons=args.horizons,
        output_mode="sequence",
    )
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No valid samples. Check data quality.")
        return

    # Class distribution
    dist = dataset.get_class_distribution()
    print(f"Class distribution: {json.dumps(dist, indent=2)}")

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_prediction_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_prediction_batch,
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # ── Initialize model ──────────────────────────────────────────────────
    model = KLineTransformerPredictor(
        n_features=9,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        num_classes=5,
        max_seq_len=args.window_size + 10,
        predict_horizons=args.horizons,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Class weights for imbalanced data
    class_weights = dataset.compute_class_weights(args.horizons[0]).to(device)
    criterion = TransformerPredictionLoss(
        num_classes=5,
        class_weight=class_weights,
        regression_weight=0.3,
        label_smoothing=0.1,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98)
    )

    # Warmup + cosine annealing scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device == "cuda") else None

    # ── Training loop ─────────────────────────────────────────────────────
    output_dir = settings.MODELS_DIR / "predictors"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15  # Early stopping patience
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={lr_now:.2e})")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.horizons, scaler=scaler,
        )
        val_metrics = validate(model, val_loader, criterion, device, args.horizons)
        scheduler.step()

        # Log
        h0 = args.horizons[0]
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracies"][h0])
        history["val_acc"].append(val_metrics["accuracies"][h0])

        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Cls: {train_metrics['cls_loss']:.4f}, "
            f"Reg: {train_metrics['reg_loss']:.4f}"
        )
        acc_str = ", ".join(
            f"T+{h}: {train_metrics['accuracies'][h]:.1f}%"
            for h in args.horizons
        )
        print(f"  Train Acc - {acc_str}")

        print(f"  Val   - Loss: {val_metrics['loss']:.4f}")
        acc_str = ", ".join(
            f"T+{h}: {val_metrics['accuracies'][h]:.1f}%"
            for h in args.horizons
        )
        print(f"  Val   Acc - {acc_str}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            model_path = output_dir / f"{model_name}.pt"
            model.save(str(model_path))
            print(f"  ** Saved best model (val_loss: {val_metrics['loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Save training history
    history_path = output_dir / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save training config
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "symbol": args.symbol,
                "market": args.market,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "layers": args.layers,
                "dim_ff": args.dim_ff,
                "window_size": args.window_size,
                "horizons": args.horizons,
                "best_val_loss": best_val_loss,
                "total_epochs": epoch + 1,
                "total_params": n_params,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / f'{model_name}.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
