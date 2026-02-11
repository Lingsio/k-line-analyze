"""
Quantile-based flat-sample filtering for stock prediction.

Removes training samples where |return| falls in the middle of the
historical distribution (ambiguous direction). Per-stock thresholds
adapt to each stock's volatility profile.
"""

import numpy as np
import pandas as pd


class QuantileThresholdCalculator:
    """
    Pre-computes per-stock return thresholds using quantile analysis.

    Samples with |return| below the threshold are considered "flat"
    (ambiguous direction) and filtered out during training.

    Only applied to training data; val/test keep all samples for
    unbiased evaluation.
    """

    def __init__(self, filter_quantile=0.35, prediction_horizon=5):
        """
        Args:
            filter_quantile: Fraction of samples to filter (0.0-0.5).
                             0.35 removes middle 35% of |return| distribution.
            prediction_horizon: Days ahead used for return calculation.
        """
        self.filter_quantile = filter_quantile
        self.prediction_horizon = prediction_horizon
        self.stock_thresholds = {}

    def fit(self, df, ticker):
        """
        Compute filtering threshold for one stock using its price history.

        Must be called on training-period data only to avoid leakage.

        Args:
            df: DataFrame with 'Close' column, sorted by date.
            ticker: Stock identifier string.
        """
        closes = df['Close'].values.astype(np.float64)
        n = len(closes)

        if n <= self.prediction_horizon + 10:
            self.stock_thresholds[ticker] = 0.005
            return

        # h-day forward returns (same calculation as in dataset._load_data)
        future = closes[self.prediction_horizon:]
        current = closes[:n - self.prediction_horizon]
        returns = (future - current) / np.where(current > 0, current, 1e-8)

        abs_returns = np.abs(returns)
        abs_returns = abs_returns[np.isfinite(abs_returns)]

        if len(abs_returns) < 20:
            self.stock_thresholds[ticker] = 0.005
            return

        # Threshold = quantile of |return| distribution
        # E.g. filter_quantile=0.35 → threshold = 35th percentile of |return|
        # Samples with |return| < threshold are "too flat" to learn from
        threshold = float(np.quantile(abs_returns, self.filter_quantile))

        # Clamp to reasonable range
        threshold = max(0.002, min(0.08, threshold))

        self.stock_thresholds[ticker] = threshold

    def should_filter(self, ticker, return_value):
        """
        Check if a sample should be excluded from training.

        Args:
            ticker: Stock identifier.
            return_value: The h-day forward return for this sample.

        Returns:
            True if the sample is in the "flat" zone and should be skipped.
        """
        threshold = self.stock_thresholds.get(ticker, 0.005)
        return abs(return_value) < threshold

    def get_threshold(self, ticker):
        """Get the pre-computed threshold for a stock."""
        return self.stock_thresholds.get(ticker, 0.005)

    def summary(self):
        """Print filtering statistics."""
        if not self.stock_thresholds:
            print("No thresholds computed yet.")
            return
        vals = list(self.stock_thresholds.values())
        print(f"Quantile filter (middle {self.filter_quantile:.0%} removed):")
        print(f"  Stocks: {len(vals)}")
        print(f"  Threshold range: {min(vals):.4f} - {max(vals):.4f}")
        print(f"  Median threshold: {np.median(vals):.4f}")
