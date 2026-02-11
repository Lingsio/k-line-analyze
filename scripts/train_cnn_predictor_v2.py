"""
Training script for CNN stock predictor v2 (k-line-net-mc).

v2 changes:
- 10-day window (each candle ~25px wide at 256x256)
- 256x256 image resolution
- Time-based train/val split (no data leakage)
- Default horizons: T+1, T+3, T+5

Usage:
    # All-stock training
    python scripts/train_cnn_predictor_v2.py --mode all --markets us tw cn

    # Single-stock training
    python scripts/train_cnn_predictor_v2.py --mode single --symbol AAPL --market us

    # Custom window (e.g., 20 days at 256x256 → ~12px per candle)
    python scripts/train_cnn_predictor_v2.py --mode single --symbol 2330 --market tw \
        --window-size 20 --horizons 1 3 5 10
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm

from models.cnn_predictor_v2 import KLineCNNPredictorV2
from models.cnn_encoder import create_default_transforms
from models.prediction_dataset_v2 import (
    StockPredictionDatasetV2,
    collate_prediction_batch,
    load_stock_data_for_training,
    time_based_split,
)
from models.transformer_predictor import TransformerPredictionLoss
from core.config import settings


STOCK_LISTS = {
    "us": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
        "NFLX", "INTC", "JPM", "BAC", "V", "MA", "JNJ", "PFE", "UNH",
        "PG", "KO", "WMT", "XOM", "CVX", "BA", "DIS", "CRM", "CSCO",
    ],
}


def train_epoch(model, dataloader, criterion, optimizer, device, horizons):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = {h: 0 for h in horizons}
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        returns = batch["returns"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss_total = torch.tensor(0.0, device=device)
        batch_cls_loss = 0
        batch_reg_loss = 0

        for i, h in enumerate(horizons):
            key = f"t{h}"
            loss_dict = criterion(
                outputs[f"cls_{key}"], labels[:, i],
                outputs[f"reg_{key}"], returns[:, i],
            )
            loss_total = loss_total + loss_dict["total"]
            batch_cls_loss += loss_dict["cls_loss"].item()
            batch_reg_loss += loss_dict["reg_loss"].item()

            preds = outputs[f"cls_{key}"].argmax(dim=-1)
            correct[h] += (preds == labels[:, i]).sum().item()

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_total.item()
        total_cls_loss += batch_cls_loss
        total_reg_loss += batch_reg_loss
        total += labels.size(0)

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "reg_loss": total_reg_loss / n,
        "accuracies": {h: correct[h] / total * 100 for h in horizons},
    }


def validate(model, dataloader, criterion, device, horizons):
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
                    outputs[f"cls_{key}"], labels[:, i],
                    outputs[f"reg_{key}"], returns[:, i],
                )
                loss_total = loss_total + loss_dict["total"]
                preds = outputs[f"cls_{key}"].argmax(dim=-1)
                correct[h] += (preds == labels[:, i]).sum().item()

            total_loss += loss_total.item()
            total += labels.size(0)

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "accuracies": {h: correct[h] / total * 100 for h in horizons},
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN stock predictor v2 (10-day/256px)")
    parser.add_argument("--mode", choices=["all", "single"], default="all")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--market", type=str, default="us")
    parser.add_argument("--markets", nargs="+", default=["us"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=10,
                        help="Candle days per image (default: 10)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="K-line image resolution (default: 256)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5],
                        help="Prediction horizons in days")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--freeze-backbone", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (datetime.now() - timedelta(days=365 * 8)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("K-Line CNN Predictor v2 (10-day / 256x256)")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {device}")
    print(f"Window: {args.window_size} days → {args.image_size}x{args.image_size} image")
    print(f"  Each candle ≈ {args.image_size // args.window_size}px wide")
    print(f"Horizons: {args.horizons}")
    print(f"Date range: {start_date} ~ {end_date}")
    print(f"Train/Val split: time-based ({args.val_ratio:.0%} val)")

    # ── Load data ─────────────────────────────────────────────────────────
    if args.mode == "single":
        if not args.symbol:
            print("ERROR: --symbol required for single mode")
            return
        print(f"Symbol: {args.symbol} ({args.market.upper()})")
        data_list = load_stock_data_for_training(
            [args.symbol], args.market, start_date, end_date,
        )
        model_name = f"cnn_v2_{args.market}_{args.symbol}"
    else:
        all_data = []
        for market in args.markets:
            symbols = STOCK_LISTS.get(market, [])
            print(f"Loading {market.upper()} ({len(symbols)} symbols)...")
            data = load_stock_data_for_training(symbols, market, start_date, end_date)
            all_data.extend(data)
            print(f"  Loaded {len(data)} stocks")
        data_list = all_data
        model_name = "cnn_v2_universal"

    if not data_list:
        print("ERROR: No data loaded.")
        return

    print(f"Total stocks: {len(data_list)}")

    # ── Create dataset ────────────────────────────────────────────────────
    train_transform, _ = create_default_transforms()

    dataset = StockPredictionDatasetV2(
        data_list=data_list,
        window_size=args.window_size,
        predict_horizons=args.horizons,
        output_mode="image",
        image_size=args.image_size,
        transform=train_transform,
    )
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No valid samples.")
        return

    dist = dataset.get_class_distribution()
    print(f"Class distribution: {json.dumps(dist, indent=2)}")

    # Time-based split (no data leakage)
    train_subset, val_subset = time_based_split(dataset, args.val_ratio)
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)} (time-split)")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_prediction_batch,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_prediction_batch,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = KLineCNNPredictorV2(
        num_classes=2, pretrained=True, predict_horizons=args.horizons,
    ).to(device)

    class_weights = dataset.compute_class_weights(args.horizons[0]).to(device)
    criterion = TransformerPredictionLoss(
        num_classes=2, class_weight=class_weights,
        regression_weight=0.3, label_smoothing=0.1,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ── Train ─────────────────────────────────────────────────────────────
    output_dir = settings.MODELS_DIR / "predictors"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if args.freeze_backbone > 0:
        model.freeze_backbone()
        print(f"Backbone frozen for first {args.freeze_backbone} epochs")

    for epoch in range(args.epochs):
        if epoch == args.freeze_backbone and args.freeze_backbone > 0:
            model.unfreeze_backbone()
            print("Backbone unfrozen")

        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.2e})")

        t = train_epoch(model, train_loader, criterion, optimizer, device, args.horizons)
        v = validate(model, val_loader, criterion, device, args.horizons)
        scheduler.step()

        h0 = args.horizons[0]
        history["train_loss"].append(t["loss"])
        history["val_loss"].append(v["loss"])
        history["train_acc"].append(t["accuracies"][h0])
        history["val_acc"].append(v["accuracies"][h0])

        print(f"  Train - Loss: {t['loss']:.4f}, Cls: {t['cls_loss']:.4f}, Reg: {t['reg_loss']:.4f}")
        train_acc_str = ', '.join(f"T+{h}: {t['accuracies'][h]:.1f}%" for h in args.horizons)
        print(f"  Train Acc - {train_acc_str}")
        print(f"  Val   - Loss: {v['loss']:.4f}")
        val_acc_str = ', '.join(f"T+{h}: {v['accuracies'][h]:.1f}%" for h in args.horizons)
        print(f"  Val   Acc - {val_acc_str}")

        if v["loss"] < best_val_loss:
            best_val_loss = v["loss"]
            model.save(str(output_dir / f"{model_name}.pt"))
            print(f"  ** Saved best model (val_loss: {v['loss']:.4f})")

    # Save history
    with open(output_dir / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model: {output_dir / f'{model_name}.pt'}")


if __name__ == "__main__":
    main()
