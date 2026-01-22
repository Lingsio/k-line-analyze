"""
Training script for the K-line pattern encoder.

Uses self-supervised learning with Triplet Loss to train the CNN encoder
to generate meaningful embeddings for K-line patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

from app.models.cnn_encoder import (
    KLineEncoder,
    TripletLoss,
    KLineDataset,
    create_default_transforms,
)
from app.services.preprocessor import KLinePreprocessor
from app.config import settings


def load_sample_data(data_dir: Path, max_samples: int = 10000):
    """
    Load sample K-line images for training.

    In production, this would load pre-generated images from disk.
    For demo, we generate synthetic data.
    """
    print("Generating synthetic training data...")

    preprocessor = KLinePreprocessor()
    images = []

    # Generate synthetic K-line patterns
    for i in tqdm(range(max_samples), desc="Generating samples"):
        # Random OHLCV data (60 days)
        n_days = 60

        # Generate random walk for close prices
        returns = np.random.normal(0, 0.02, n_days)
        close = 100 * np.cumprod(1 + returns)

        # Generate OHLC from close
        data = {
            'open': close * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'high': close * (1 + np.abs(np.random.normal(0.005, 0.005, n_days))),
            'low': close * (1 - np.abs(np.random.normal(0.005, 0.005, n_days))),
            'close': close,
            'volume': np.random.uniform(1e6, 1e8, n_days),
        }

        import pandas as pd
        df = pd.DataFrame(data)

        # Normalize and render
        normalized = preprocessor.normalize(df)
        image = preprocessor.to_kline_image(normalized, image_size=128)

        images.append(image)

    return images


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for anchor, positive, negative in tqdm(dataloader, desc="Training"):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        # Compute loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Configuration
    config = {
        'embedding_dim': 256,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 50,
        'margin': 0.3,
        'num_samples': 10000,
        'val_split': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("=" * 60)
    print("K-Line Pattern Encoder Training")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Embedding dimension: {config['embedding_dim']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print("=" * 60)

    # Load data
    data_dir = settings.DATA_DIR / 'processed'
    images = load_sample_data(data_dir, config['num_samples'])

    # Split into train/val
    split_idx = int(len(images) * (1 - config['val_split']))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

    # Create datasets
    train_transform, _ = create_default_transforms()

    train_dataset = KLineDataset(train_images, transform=train_transform)
    val_dataset = KLineDataset(val_images, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    model = KLineEncoder(embedding_dim=config['embedding_dim'], pretrained=True)
    model = model.to(config['device'])

    # Loss and optimizer
    criterion = TripletLoss(margin=config['margin'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        val_loss = validate(model, val_loader, criterion, config['device'])

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = settings.MODELS_DIR / 'encoder.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            print(f"Saved best model (val_loss: {val_loss:.4f})")

    # Save training history
    history_path = settings.MODELS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {settings.MODELS_DIR / 'encoder.pt'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
