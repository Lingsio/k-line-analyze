"""
Feature extraction service combining preprocessing and CNN encoding.

Provides a unified interface for converting K-line data into embeddings.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd

from app.config import settings
from app.services.preprocessor import KLinePreprocessor
from app.models.cnn_encoder import KLineEncoder, create_default_transforms


class FeatureExtractor:
    """
    Feature extraction pipeline for K-line patterns.

    Combines preprocessing, image generation, and CNN encoding
    into a single service.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize feature extractor.

        Args:
            model_path: Path to trained model weights
            device: Device for inference ('cpu' or 'cuda')
        """
        self.device = device
        self.preprocessor = KLinePreprocessor()

        # Initialize encoder
        self.encoder = KLineEncoder(
            embedding_dim=settings.EMBEDDING_DIM,
            pretrained=True,
        )

        # Load trained weights if available
        if model_path and Path(model_path).exists():
            self.encoder.load(model_path, device=device)
        elif (settings.MODELS_DIR / "encoder.pt").exists():
            self.encoder.load(str(settings.MODELS_DIR / "encoder.pt"), device=device)

        self.encoder.to(device)
        self.encoder.eval()

        # Get transforms
        _, self.transform = create_default_transforms()

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        include_volume: bool = True,
    ) -> np.ndarray:
        """
        Extract embedding from a DataFrame of OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
            include_volume: Whether to include volume in the image

        Returns:
            Embedding vector (1D numpy array)
        """
        # Normalize
        normalized = self.preprocessor.normalize(df)

        # Generate image
        image = self.preprocessor.to_kline_image(normalized, include_volume=include_volume)

        # Transform for model
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.encoder(tensor)

        return embedding.cpu().numpy().flatten()

    def extract_batch(
        self,
        dataframes: List[pd.DataFrame],
        include_volume: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract embeddings for multiple DataFrames.

        Args:
            dataframes: List of OHLCV DataFrames
            include_volume: Whether to include volume
            batch_size: Batch size for inference

        Returns:
            Numpy array of embeddings (n_samples x embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(dataframes), batch_size):
            batch_dfs = dataframes[i : i + batch_size]

            # Generate images
            images = []
            for df in batch_dfs:
                normalized = self.preprocessor.normalize(df)
                image = self.preprocessor.to_kline_image(normalized, include_volume=include_volume)
                tensor = self.transform(image)
                images.append(tensor)

            # Stack and encode
            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                embeddings = self.encoder(batch_tensor)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def extract_from_windows(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        step: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Extract embeddings for all sliding windows in a DataFrame.

        Args:
            df: Full OHLCV DataFrame
            window_size: Window size
            step: Step between windows

        Returns:
            List of dicts with embeddings and metadata
        """
        windows = self.preprocessor.create_windows(df, window_size, step)
        results = []

        for window_df, start_idx, end_idx in windows:
            embedding = self.extract_from_dataframe(window_df)

            results.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_date": str(window_df.index[0]) if hasattr(window_df.index[0], "strftime") else str(window_df.index[0]),
                "end_date": str(window_df.index[-1]) if hasattr(window_df.index[-1], "strftime") else str(window_df.index[-1]),
                "embedding": embedding,
            })

        return results

    def extract_multi_scale(
        self,
        df: pd.DataFrame,
        window_sizes: List[int] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Extract embeddings at multiple time scales.

        Args:
            df: Full OHLCV DataFrame
            window_sizes: List of window sizes

        Returns:
            Dict mapping window_size to embedding
        """
        window_sizes = window_sizes or settings.DEFAULT_WINDOW_SIZES
        results = {}

        for size in window_sizes:
            if len(df) >= size:
                # Use most recent window
                window_df = df.iloc[-size:]
                results[size] = self.extract_from_dataframe(window_df)

        return results


class BatchProcessor:
    """
    Batch processor for building the search index from historical data.
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        """
        Initialize batch processor.

        Args:
            feature_extractor: FeatureExtractor instance
        """
        self.extractor = feature_extractor

    async def process_stock(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
        window_size: int = 60,
        step: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Process a single stock to extract all window embeddings.

        Args:
            symbol: Stock symbol
            market: Market identifier
            start_date: Start date
            end_date: End date
            window_size: Window size
            step: Step between windows

        Returns:
            List of window embeddings with metadata
        """
        from app.services.data_fetcher import DataFetcher

        fetcher = DataFetcher()
        df = await fetcher.fetch_ohlcv(
            symbol=symbol,
            market=market,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or len(df) < window_size:
            return []

        windows = self.extractor.extract_from_windows(df, window_size, step)

        # Add symbol and market metadata
        for w in windows:
            w["symbol"] = symbol
            w["market"] = market

        return windows

    async def process_market(
        self,
        market: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        window_size: int = 60,
        step: int = 5,
        callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Process all stocks in a market.

        Args:
            market: Market identifier
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            window_size: Window size
            step: Step between windows
            callback: Optional progress callback

        Returns:
            List of all window embeddings
        """
        all_results = []

        for i, symbol in enumerate(symbols):
            try:
                results = await self.process_stock(
                    symbol=symbol,
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                    window_size=window_size,
                    step=step,
                )
                all_results.extend(results)

                if callback:
                    callback(i + 1, len(symbols), symbol)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        return all_results
