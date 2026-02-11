"""
Advanced Labeling Strategies for Stock Prediction

Implements sophisticated labeling methods based on:
1. Volatility-adjusted thresholds
2. Quantile-based adaptive thresholds
3. Regime detection (trending vs ranging)
4. Multi-horizon consensus
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = 1
    TRENDING_DOWN = 2
    RANGING = 3
    HIGH_VOLATILITY = 4
    LOW_VOLATILITY = 5


class AdvancedLabeler:
    """
    Advanced stock movement labeler with multiple strategies.
    
    Features:
    - Volatility regime detection
    - Adaptive threshold based on market conditions
    - Quantile-based labeling to handle class imbalance
    - Multi-horizon consensus labeling
    """
    
    def __init__(
        self,
        strategy: Literal['volatility', 'quantile', 'regime', 'consensus'] = 'volatility',
        prediction_horizon: int = 5,
        num_classes: int = 2,
        flat_threshold_quantile: float = 0.3,
    ):
        """
        Args:
            strategy: Labeling strategy to use
            prediction_horizon: Days ahead to predict
            num_classes: 2 (up/down) or 3 (up/flat/down)
            flat_threshold_quantile: Quantile for flat region (0-1)
        """
        self.strategy = strategy
        self.prediction_horizon = prediction_horizon
        self.num_classes = num_classes
        self.flat_threshold_quantile = flat_threshold_quantile
        
    def compute_labels(
        self,
        df: pd.DataFrame,
        current_idx: int,
        history_window: int = 60,
    ) -> Tuple[int, dict]:
        """
        Compute label for a given position.
        
        Args:
            df: DataFrame with OHLCV data
            current_idx: Current position index
            history_window: Window for computing statistics
        
        Returns:
            label: Class label (0=down, 1=flat, 2=up)
            info: Dict with additional information
        """
        # Get future price
        future_idx = min(current_idx + self.prediction_horizon, len(df) - 1)
        current_close = df.iloc[current_idx]['Close']
        future_close = df.iloc[future_idx]['Close']
        
        # Compute return
        ret = (future_close - current_close) / current_close
        
        # Get historical window
        start_idx = max(0, current_idx - history_window)
        hist_df = df.iloc[start_idx:current_idx]
        
        if self.strategy == 'volatility':
            return self._volatility_label(ret, hist_df)
        elif self.strategy == 'quantile':
            return self._quantile_label(ret, hist_df)
        elif self.strategy == 'regime':
            return self._regime_label(ret, hist_df, current_close)
        elif self.strategy == 'consensus':
            return self._consensus_label(df, current_idx)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _volatility_label(self, ret: float, hist_df: pd.DataFrame) -> Tuple[int, dict]:
        """
        Volatility-adjusted labeling.
        
        Threshold scales with historical volatility:
        - Higher volatility -> larger threshold
        - Lower volatility -> smaller threshold
        """
        # Compute historical volatility
        returns = hist_df['Close'].pct_change().dropna()
        
        if len(returns) < 10:
            vol = 0.02  # Default 2%
        else:
            # Use median absolute deviation for robustness
            vol = returns.mad() * 1.4826  # Convert MAD to std
            if vol < 0.005:
                vol = 0.005
        
        # Scale threshold by prediction horizon and volatility
        # threshold = k * vol * sqrt(horizon)
        k = 0.5  # Tuning parameter
        threshold = k * vol * np.sqrt(self.prediction_horizon)
        
        # Clamp threshold to reasonable range
        threshold = max(0.002, min(0.05, threshold))
        
        # Assign label
        if self.num_classes == 3:
            if ret > threshold:
                label = 2  # Up
            elif ret < -threshold:
                label = 0  # Down
            else:
                label = 1  # Flat
        else:
            label = 1 if ret > 0 else 0
        
        info = {
            'return': ret,
            'threshold': threshold,
            'volatility': vol,
            'strategy': 'volatility',
        }
        
        return label, info
    
    def _quantile_label(self, ret: float, hist_df: pd.DataFrame) -> Tuple[int, dict]:
        """
        Quantile-based adaptive labeling.
        
        Uses historical return distribution to define thresholds:
        - Top/bottom quantiles define up/down
        - Middle region is flat (optional)
        """
        returns = hist_df['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            # Fall back to simple threshold
            threshold = 0.005
        else:
            # Use quantiles to define thresholds
            up_threshold = returns.quantile(1 - self.flat_threshold_quantile / 2)
            down_threshold = returns.quantile(self.flat_threshold_quantile / 2)
            threshold = (up_threshold - down_threshold) / 2
        
        if self.num_classes == 3:
            # Use asymmetric thresholds based on actual quantiles
            returns_sorted = sorted(returns.dropna())
            n = len(returns_sorted)
            
            if n > 20:
                up_idx = int(n * (1 - self.flat_threshold_quantile / 2))
                down_idx = int(n * self.flat_threshold_quantile / 2)
                up_thresh = returns_sorted[up_idx]
                down_thresh = returns_sorted[down_idx]
            else:
                up_thresh = threshold
                down_thresh = -threshold
            
            if ret > up_thresh:
                label = 2
            elif ret < down_thresh:
                label = 0
            else:
                label = 1
        else:
            label = 1 if ret > 0 else 0
        
        info = {
            'return': ret,
            'threshold': threshold,
            'up_threshold': up_thresh if self.num_classes == 3 else threshold,
            'down_threshold': down_thresh if self.num_classes == 3 else -threshold,
            'strategy': 'quantile',
        }
        
        return label, info
    
    def _regime_label(
        self, 
        ret: float, 
        hist_df: pd.DataFrame,
        current_price: float
    ) -> Tuple[int, dict]:
        """
        Regime-aware labeling.
        
        Detects market regime and adjusts labeling accordingly:
        - Trending markets: require larger moves to trigger signals
        - Ranging markets: smaller thresholds, focus on mean reversion
        - High volatility: widen thresholds
        """
        # Compute regime indicators
        closes = hist_df['Close'].values
        
        if len(closes) < 20:
            regime = MarketRegime.RANGING
            trend_strength = 0
        else:
            # Trend detection using linear regression
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)
            
            # R-squared as trend strength
            pred = slope * x + intercept
            ss_res = np.sum((closes - pred) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trend_strength = r_squared
            
            # Volatility regime
            returns = hist_df['Close'].pct_change().dropna()
            vol = returns.std()
            vol_percentile = (returns.abs().quantile(0.9))
            
            # Classify regime
            if trend_strength > 0.6:  # Strong trend
                if slope > 0:
                    regime = MarketRegime.TRENDING_UP
                else:
                    regime = MarketRegime.TRENDING_DOWN
            elif vol_percentile > 0.03:  # High volatility
                regime = MarketRegime.HIGH_VOLATILITY
            elif vol_percentile < 0.01:  # Low volatility
                regime = MarketRegime.LOW_VOLATILITY
            else:
                regime = MarketRegime.RANGING
        
        # Adjust threshold based on regime
        base_threshold = 0.005
        
        if regime == MarketRegime.TRENDING_UP:
            # In uptrend, require stronger move for up signal, weaker for down
            up_threshold = base_threshold * 1.5
            down_threshold = base_threshold * 0.7
        elif regime == MarketRegime.TRENDING_DOWN:
            up_threshold = base_threshold * 0.7
            down_threshold = base_threshold * 1.5
        elif regime == MarketRegime.HIGH_VOLATILITY:
            up_threshold = base_threshold * 1.3
            down_threshold = base_threshold * 1.3
        elif regime == MarketRegime.LOW_VOLATILITY:
            up_threshold = base_threshold * 0.6
            down_threshold = base_threshold * 0.6
        else:  # Ranging
            up_threshold = base_threshold
            down_threshold = base_threshold
        
        # Assign label
        if self.num_classes == 3:
            if ret > up_threshold:
                label = 2
            elif ret < -down_threshold:
                label = 0
            else:
                label = 1
        else:
            label = 1 if ret > 0 else 0
        
        info = {
            'return': ret,
            'regime': regime.name,
            'trend_strength': trend_strength,
            'up_threshold': up_threshold,
            'down_threshold': down_threshold,
            'strategy': 'regime',
        }
        
        return label, info
    
    def _consensus_label(self, df: pd.DataFrame, current_idx: int) -> Tuple[int, dict]:
        """
        Multi-horizon consensus labeling.
        
        Looks at multiple prediction horizons and requires consensus
        to reduce noise and focus on high-confidence signals.
        """
        horizons = [1, 3, 5, 10]
        labels = []
        returns = []
        
        current_close = df.iloc[current_idx]['Close']
        
        for h in horizons:
            future_idx = min(current_idx + h, len(df) - 1)
            future_close = df.iloc[future_idx]['Close']
            ret = (future_close - current_close) / current_close
            returns.append(ret)
            labels.append(1 if ret > 0 else 0)
        
        # Require consensus across horizons
        up_votes = sum(labels)
        down_votes = len(labels) - up_votes
        
        # Consensus threshold
        consensus_ratio = 0.6
        min_votes = int(len(labels) * consensus_ratio)
        
        if up_votes >= min_votes:
            label = 2 if self.num_classes == 3 else 1
        elif down_votes >= min_votes:
            label = 0
        else:
            label = 1 if self.num_classes == 3 else (1 if np.mean(returns) > 0 else 0)
        
        info = {
            'returns': returns,
            'up_votes': up_votes,
            'down_votes': down_votes,
            'consensus_ratio': max(up_votes, down_votes) / len(labels),
            'strategy': 'consensus',
        }
        
        return label, info


class SmartThresholdCalculator:
    """
    Smart threshold calculator that adapts to each stock's characteristics.
    
    Pre-computes per-stock thresholds based on historical data
    to avoid data leakage during training.
    """
    
    def __init__(self, prediction_horizon: int = 5):
        self.prediction_horizon = prediction_horizon
        self.stock_thresholds = {}
        self.stock_stats = {}
    
    def fit(self, df: pd.DataFrame, ticker: str):
        """
        Compute thresholds for a stock using historical data.
        
        Should be called on training data only to avoid leakage.
        """
        returns = df['Close'].pct_change().dropna()
        
        # Compute statistics
        mean_ret = returns.mean()
        std_ret = returns.std()
        median_ret = returns.median()
        
        # Volatility at different horizons
        horizon_returns = df['Close'].pct_change(self.prediction_horizon).dropna()
        horizon_vol = horizon_returns.std()
        
        # Adaptive threshold
        # Based on: threshold should filter out ~30% of samples as "flat"
        abs_returns = horizon_returns.abs()
        threshold = abs_returns.quantile(0.3)
        
        # Ensure reasonable bounds
        threshold = max(0.003, min(0.05, threshold))
        
        self.stock_thresholds[ticker] = threshold
        self.stock_stats[ticker] = {
            'mean_return': mean_ret,
            'std_return': std_ret,
            'median_return': median_ret,
            'horizon_volatility': horizon_vol,
            'quantile_30': threshold,
        }
    
    def get_threshold(self, ticker: str) -> float:
        """Get pre-computed threshold for a stock."""
        return self.stock_thresholds.get(ticker, 0.005)
    
    def get_stats(self, ticker: str) -> dict:
        """Get pre-computed statistics for a stock."""
        return self.stock_stats.get(ticker, {})


def filter_extreme_samples(samples: list, extreme_quantile: float = 0.01) -> list:
    """
    Filter out extreme return samples that may be outliers.
    
    Args:
        samples: List of sample dicts with 'raw_return' key
        extreme_quantile: Quantile threshold for extremes
    
    Returns:
        Filtered list of samples
    """
    returns = [s['raw_return'] for s in samples]
    
    lower = np.quantile(returns, extreme_quantile)
    upper = np.quantile(returns, 1 - extreme_quantile)
    
    filtered = [
        s for s in samples 
        if lower <= s['raw_return'] <= upper
    ]
    
    return filtered


def balance_classes(samples: list, method: Literal['undersample', 'oversample'] = 'undersample') -> list:
    """
    Balance class distribution.
    
    Args:
        samples: List of sample dicts with 'label' key
        method: 'undersample' or 'oversample'
    
    Returns:
        Balanced list of samples
    """
    from collections import defaultdict
    
    # Group by label
    by_label = defaultdict(list)
    for s in samples:
        by_label[s['label']].append(s)
    
    # Find min/max counts
    counts = {label: len(group) for label, group in by_label.items()}
    
    if method == 'undersample':
        target = min(counts.values())
        balanced = []
        for label, group in by_label.items():
            # Random sample without replacement
            indices = np.random.choice(len(group), size=target, replace=False)
            balanced.extend([group[i] for i in indices])
    else:  # oversample
        target = max(counts.values())
        balanced = []
        for label, group in by_label.items():
            # Random sample with replacement
            if len(group) < target:
                indices = np.random.choice(len(group), size=target, replace=True)
            else:
                indices = np.random.choice(len(group), size=target, replace=False)
            balanced.extend([group[i] for i in indices])
    
    return balanced
