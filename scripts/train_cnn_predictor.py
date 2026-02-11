"""
Training script for CNN-based stock predictor (k-line-net-mc).

Supports two training modes:
1. All-stock (universal): Train on multiple stocks across markets
2. Single-stock: Train a specialized model for one specific ticker

Usage:
    # All-stock training
    python scripts/train_cnn_predictor.py --mode all --markets us tw cn

    # Single-stock training
    python scripts/train_cnn_predictor.py --mode single --symbol AAPL --market us

    # Single-stock with custom params
    python scripts/train_cnn_predictor.py --mode single --symbol 2330 --market tw \
        --epochs 100 --lr 5e-5 --horizons 1 5 10 20
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

from app.models.cnn_predictor import KLineCNNPredictor
from app.models.cnn_encoder import create_default_transforms
from app.models.prediction_dataset import (
    StockPredictionDataset,
    collate_prediction_batch,
    load_stock_data_for_training,
)
from app.models.transformer_predictor import TransformerPredictionLoss
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


def train_epoch(model, dataloader, criterion, optimizer, device, horizons):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = {h: 0 for h in horizons}
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)  # (batch, n_horizons)
        returns = batch["returns"].to(device)  # (batch, n_horizons)

        optimizer.zero_grad()
        outputs = model(images)

        # Sum loss over all horizons
        loss_total = torch.tensor(0.0, device=device)
        batch_cls_loss = 0
        batch_reg_loss = 0

        for i, h in enumerate(horizons):
            key = f"t{h}"
            loss_dict = criterion(
                outputs[f"cls_{key}"],
                labels[:, i],
                outputs[f"reg_{key}"],
                returns[:, i],
            )
            loss_total = loss_total + loss_dict["total"]
            batch_cls_loss += loss_dict["cls_loss"].item()
            batch_reg_loss += loss_dict["reg_loss"].item()

            # Accuracy
            preds = outputs[f"cls_{key}"].argmax(dim=-1)
            correct[h] += (preds == labels[:, i]).sum().item()

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_total.item()
        total_cls_loss += batch_cls_loss
        total_reg_loss += batch_reg_loss
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
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            returns = batch["returns"].to(device)

            outputs = model(images)

            loss_total = torch.tensor(0.0, device=device)
            for i, h in enumerate(horizons):
                key = f"t{h}"
                loss_dict = criterion(
                    outputs[f"cls_{key}"],
                    labels[:, i],
                    outputs[f"reg_{key}"],
                    returns[:, i],
                )
                loss_total = loss_total + loss_dict["total"]

                preds = outputs[f"cls_{key}"].argmax(dim=-1)
                correct[h] += (preds == labels[:, i]).sum().item()

            total_loss += loss_total.item()
            total += labels.size(0)

    n_batches = len(dataloader)
    accuracies = {h: correct[h] / total * 100 for h in horizons}
    return {
        "loss": total_loss / n_batches,
        "accuracies": accuracies,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN stock predictor (k-line-net-mc)")
    parser.add_argument("--mode", type=str, choices=["all", "single"], default="all",
                        help="Training mode: 'all' for universal, 'single' for per-stock")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Stock symbol (required for single mode)")
    parser.add_argument("--market", type=str, default="us",
                        help="Market for the symbol")
    parser.add_argument("--markets", nargs="+", default=["us"],
                        help="Markets to include in all-stock mode")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20],
                        help="Prediction horizons in days")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--freeze-backbone", type=int, default=5,
                        help="Freeze backbone for N epochs")
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (datetime.now() - timedelta(days=365 * 8)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("K-Line CNN Predictor Training (k-line-net-mc)")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {device}")
    print(f"Horizons: {args.horizons}")
    print(f"Window size: {args.window_size}")
    print(f"Date range: {start_date} ~ {end_date}")

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
        model_name = f"cnn_predictor_{args.market}_{args.symbol}"
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
        model_name = "cnn_predictor_universal"

    if not data_list:
        print("ERROR: No data loaded. Exiting.")
        return

    print(f"Total stocks loaded: {len(data_list)}")

    # ── Create dataset ────────────────────────────────────────────────────
    train_transform, _ = create_default_transforms()

    dataset = StockPredictionDataset(
        data_list=data_list,
        window_size=args.window_size,
        predict_horizons=args.horizons,
        output_mode="image",
        transform=train_transform,
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
        num_workers=2, pin_memory=True, collate_fn=collate_prediction_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_prediction_batch,
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # ── Initialize model ──────────────────────────────────────────────────
    model = KLineCNNPredictor(
        num_classes=5,
        pretrained=True,
        predict_horizons=args.horizons,
    ).to(device)

    # Class weights for imbalanced data
    class_weights = dataset.compute_class_weights(args.horizons[0]).to(device)
    criterion = TransformerPredictionLoss(
        num_classes=5,
        class_weight=class_weights,
        regression_weight=0.3,
        label_smoothing=0.1,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # ── Training loop ─────────────────────────────────────────────────────
    output_dir = settings.MODELS_DIR / "predictors"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Freeze backbone initially
    if args.freeze_backbone > 0:
        model.freeze_backbone()
        print(f"Backbone frozen for first {args.freeze_backbone} epochs")

    for epoch in range(args.epochs):
        # Unfreeze backbone after N epochs
        if epoch == args.freeze_backbone and args.freeze_backbone > 0:
            model.unfreeze_backbone()
            print("Backbone unfrozen")

        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.2e})")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, args.horizons
        )
        val_metrics = validate(
            model, val_loader, criterion, device, args.horizons
        )
        scheduler.step()

        # Log metrics
        h0 = args.horizons[0]
        train_acc = train_metrics["accuracies"][h0]
        val_acc = val_metrics["accuracies"][h0]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

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
            model_path = output_dir / f"{model_name}.pt"
            model.save(str(model_path))
            print(f"  ** Saved best model (val_loss: {val_metrics['loss']:.4f})")

    # Save training history
    history_path = output_dir / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / f'{model_name}.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
