"""
Training script for Transformer stock predictor v2.

v2 changes:
- Default 10-day window (matching CNN v2)
- Default horizons T+1, T+3, T+5
- Time-based train/val split (no data leakage)
- Sequence input: 10 steps × 9 features

Usage:
    # All-stock
    python scripts/train_transformer_predictor_v2.py --mode all --markets us tw cn

    # Single-stock
    python scripts/train_transformer_predictor_v2.py --mode single --symbol 2330 --market tw

    # H20 GPU optimized
    python scripts/train_transformer_predictor_v2.py --mode all --markets us tw cn hk \
        --batch-size 256 --d-model 256 --nhead 8 --layers 6 --amp
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

from models.transformer_predictor import (
    KLineTransformerPredictor,
    TransformerPredictionLoss,
)
from models.prediction_dataset_v2 import (
    StockPredictionDatasetV2,
    collate_prediction_batch,
    load_stock_data_for_training,
    time_based_split,
)
from core.config import settings


STOCK_LISTS = {
    "us": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
        "NFLX", "INTC", "JPM", "BAC", "V", "MA", "JNJ", "PFE", "UNH",
        "PG", "KO", "WMT", "XOM", "CVX", "BA", "DIS", "CRM", "CSCO",
    ],
}


def train_epoch(model, dataloader, criterion, optimizer, device, horizons, scaler=None):
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
            with torch.amp.autocast("cuda"):
                outputs = model(sequences)
                loss_total = torch.tensor(0.0, device=device)
                batch_cls = 0
                batch_reg = 0
                for i, h in enumerate(horizons):
                    key = f"t{h}"
                    ld = criterion(outputs[f"cls_{key}"], labels[:, i],
                                   outputs[f"reg_{key}"], returns[:, i])
                    loss_total = loss_total + ld["total"]
                    batch_cls += ld["cls_loss"].item()
                    batch_reg += ld["reg_loss"].item()

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
                ld = criterion(outputs[f"cls_{key}"], labels[:, i],
                               outputs[f"reg_{key}"], returns[:, i])
                loss_total = loss_total + ld["total"]
                batch_cls += ld["cls_loss"].item()
                batch_reg += ld["reg_loss"].item()

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            for i, h in enumerate(horizons):
                preds = outputs[f"cls_t{h}"].argmax(dim=-1)
                correct[h] += (preds == labels[:, i]).sum().item()

        total_loss += loss_total.item()
        total_cls_loss += batch_cls
        total_reg_loss += batch_reg
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
            sequences = batch["sequence"].to(device)
            labels = batch["labels"].to(device)
            returns = batch["returns"].to(device)

            outputs = model(sequences)
            loss_total = torch.tensor(0.0, device=device)
            for i, h in enumerate(horizons):
                key = f"t{h}"
                ld = criterion(outputs[f"cls_{key}"], labels[:, i],
                               outputs[f"reg_{key}"], returns[:, i])
                loss_total = loss_total + ld["total"]
                preds = outputs[f"cls_{key}"].argmax(dim=-1)
                correct[h] += (preds == labels[:, i]).sum().item()

            total_loss += loss_total.item()
            total += labels.size(0)

    n = len(dataloader)
    return {"loss": total_loss / n, "accuracies": {h: correct[h] / total * 100 for h in horizons}}


def main():
    parser = argparse.ArgumentParser(description="Train Transformer predictor v2 (10-day)")
    parser.add_argument("--mode", choices=["all", "single"], default="all")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--market", type=str, default="us")
    parser.add_argument("--markets", nargs="+", default=["us"])

    # Architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window-size", type=int, default=10,
                        help="Input sequence length in days (default: 10)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (datetime.now() - timedelta(days=365 * 8)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("K-Line Transformer Predictor v2 (10-day / Self-Attention)")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {device}")
    print(f"Architecture: d={args.d_model}, heads={args.nhead}, layers={args.layers}, ff={args.dim_ff}")
    print(f"Window: {args.window_size} days, Horizons: {args.horizons}")
    print(f"Date range: {start_date} ~ {end_date}")
    print(f"AMP: {args.amp}, Split: time-based ({args.val_ratio:.0%} val)")

    # ── Load data ─────────────────────────────────────────────────────────
    if args.mode == "single":
        if not args.symbol:
            print("ERROR: --symbol required")
            return
        print(f"Symbol: {args.symbol} ({args.market.upper()})")
        data_list = load_stock_data_for_training(
            [args.symbol], args.market, start_date, end_date,
        )
        model_name = f"transformer_v2_{args.market}_{args.symbol}"
    else:
        all_data = []
        for market in args.markets:
            symbols = STOCK_LISTS.get(market, [])
            print(f"Loading {market.upper()} ({len(symbols)} symbols)...")
            data = load_stock_data_for_training(symbols, market, start_date, end_date)
            all_data.extend(data)
            print(f"  Loaded {len(data)} stocks")
        data_list = all_data
        model_name = "transformer_v2_universal"

    if not data_list:
        print("ERROR: No data loaded.")
        return

    print(f"Total stocks: {len(data_list)}")

    # ── Dataset (sequence mode) ───────────────────────────────────────────
    dataset = StockPredictionDatasetV2(
        data_list=data_list,
        window_size=args.window_size,
        predict_horizons=args.horizons,
        output_mode="sequence",
    )
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No valid samples.")
        return

    dist = dataset.get_class_distribution()
    print(f"Class distribution: {json.dumps(dist, indent=2)}")

    # Time-based split
    train_subset, val_subset = time_based_split(dataset, args.val_ratio)
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)} (time-split)")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_prediction_batch,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_prediction_batch,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = KLineTransformerPredictor(
        n_features=9,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        num_classes=2,
        max_seq_len=args.window_size + 10,
        predict_horizons=args.horizons,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    class_weights = dataset.compute_class_weights(args.horizons[0]).to(device)
    criterion = TransformerPredictionLoss(
        num_classes=2, class_weight=class_weights,
        regression_weight=0.3, label_smoothing=0.1,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98))

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device == "cuda") else None

    # ── Train ─────────────────────────────────────────────────────────────
    output_dir = settings.MODELS_DIR / "predictors"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.2e})")

        t = train_epoch(model, train_loader, criterion, optimizer, device, args.horizons, scaler)
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
            patience_counter = 0
            model.save(str(output_dir / f"{model_name}.pt"))
            print(f"  ** Saved best (val_loss: {v['loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Save history & config
    with open(output_dir / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / f"{model_name}_config.json", "w") as f:
        json.dump({
            "mode": args.mode, "symbol": args.symbol, "market": args.market,
            "d_model": args.d_model, "nhead": args.nhead, "layers": args.layers,
            "dim_ff": args.dim_ff, "window_size": args.window_size,
            "horizons": args.horizons, "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1, "total_params": n_params,
        }, f, indent=2)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model: {output_dir / f'{model_name}.pt'}")


if __name__ == "__main__":
    main()
