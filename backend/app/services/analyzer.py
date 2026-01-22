"""
Pattern analysis module for computing statistics on similar K-line patterns.

Analyzes subsequent price movements of matched historical patterns
to generate predictive insights.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd


class PatternAnalyzer:
    """
    Analyzer for K-line pattern outcomes.

    Computes statistics on what happened after similar patterns occurred
    in history, providing win rates, average returns, etc.
    """

    def __init__(self):
        """Initialize analyzer."""
        self.lookahead_days = [1, 5, 10, 20]

    async def get_subsequent_returns(
        self,
        symbol: str,
        market: str,
        end_date: str,
        lookahead_days: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Get returns for periods after the pattern end date.

        Args:
            symbol: Stock symbol
            market: Market identifier
            end_date: End date of the pattern
            lookahead_days: Days to look ahead

        Returns:
            Dict with returns for each period
        """
        from app.services.data_fetcher import DataFetcher

        lookahead_days = lookahead_days or self.lookahead_days
        fetcher = DataFetcher()

        # Calculate date range for fetching future data
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        future_start = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        future_end = (end_dt + timedelta(days=max(lookahead_days) + 10)).strftime("%Y-%m-%d")

        try:
            df = await fetcher.fetch_ohlcv(
                symbol=symbol,
                market=market,
                start_date=future_start,
                end_date=future_end,
            )

            if df is None or df.empty:
                return {}

            # Get close price at pattern end
            # First, get the data up to end_date to find the closing price
            pattern_df = await fetcher.fetch_ohlcv(
                symbol=symbol,
                market=market,
                start_date=(end_dt - timedelta(days=5)).strftime("%Y-%m-%d"),
                end_date=end_date,
            )

            if pattern_df is None or pattern_df.empty:
                return {}

            base_price = pattern_df["close"].iloc[-1]

            # Calculate returns for each lookahead period
            returns = {}
            for days in lookahead_days:
                if days <= len(df):
                    future_price = df["close"].iloc[days - 1]
                    ret = (future_price - base_price) / base_price
                    returns[f"t+{days}"] = round(float(ret), 4)

            return returns

        except Exception as e:
            print(f"Error getting subsequent returns: {e}")
            return {}

    def analyze_outcomes(
        self,
        similar_patterns: List[Any],
        lookahead_days: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze outcomes of similar patterns.

        Args:
            similar_patterns: List of SimilarPattern objects or dicts
            lookahead_days: Periods to analyze

        Returns:
            AnalysisSummary as dict
        """
        lookahead_days = lookahead_days or self.lookahead_days

        if not similar_patterns:
            return self._empty_analysis()

        # Extract returns for each period
        returns_by_period = {f"t+{d}": [] for d in lookahead_days}

        for pattern in similar_patterns:
            # Handle both SimilarPattern objects and dicts
            if hasattr(pattern, "subsequent_returns"):
                sub_ret = pattern.subsequent_returns
                if hasattr(sub_ret, "t_plus_1") and sub_ret.t_plus_1 is not None:
                    returns_by_period["t+1"].append(sub_ret.t_plus_1)
                if hasattr(sub_ret, "t_plus_5") and sub_ret.t_plus_5 is not None:
                    returns_by_period["t+5"].append(sub_ret.t_plus_5)
                if hasattr(sub_ret, "t_plus_10") and sub_ret.t_plus_10 is not None:
                    returns_by_period["t+10"].append(sub_ret.t_plus_10)
                if hasattr(sub_ret, "t_plus_20") and sub_ret.t_plus_20 is not None:
                    returns_by_period["t+20"].append(sub_ret.t_plus_20)
            elif isinstance(pattern, dict):
                sub_ret = pattern.get("subsequent_returns", {})
                for period in lookahead_days:
                    key = f"t+{period}"
                    if key in sub_ret and sub_ret[key] is not None:
                        returns_by_period[key].append(sub_ret[key])

        # Calculate statistics
        stats = {}
        for period, returns in returns_by_period.items():
            if returns:
                returns = np.array(returns)
                stats[period] = {
                    "count": len(returns),
                    "win_rate": float(np.mean(returns > 0)),
                    "avg_return": float(np.mean(returns)),
                    "median_return": float(np.median(returns)),
                    "std": float(np.std(returns)),
                    "max": float(np.max(returns)),
                    "min": float(np.min(returns)),
                    "positive_avg": float(np.mean(returns[returns > 0])) if any(returns > 0) else 0,
                    "negative_avg": float(np.mean(returns[returns < 0])) if any(returns < 0) else 0,
                }

        # Calculate overall confidence based on sample size and consistency
        confidence = self._calculate_confidence(stats, len(similar_patterns))

        # Build summary
        ret_1d = returns_by_period.get("t+1", [])
        ret_5d = returns_by_period.get("t+5", [])
        ret_20d = returns_by_period.get("t+20", [])

        return {
            "total_matches": len(similar_patterns),
            "win_rate_1d": round(np.mean(np.array(ret_1d) > 0), 3) if ret_1d else 0,
            "win_rate_5d": round(np.mean(np.array(ret_5d) > 0), 3) if ret_5d else 0,
            "win_rate_20d": round(np.mean(np.array(ret_20d) > 0), 3) if ret_20d else 0,
            "avg_return_1d": round(np.mean(ret_1d), 4) if ret_1d else 0,
            "avg_return_5d": round(np.mean(ret_5d), 4) if ret_5d else 0,
            "avg_return_20d": round(np.mean(ret_20d), 4) if ret_20d else 0,
            "median_return_5d": round(np.median(ret_5d), 4) if ret_5d else 0,
            "confidence": round(confidence, 3),
            "detailed_stats": stats,
        }

    def _calculate_confidence(
        self,
        stats: Dict[str, Dict[str, float]],
        sample_size: int,
    ) -> float:
        """
        Calculate confidence score based on various factors.

        Factors:
        - Sample size (more samples = higher confidence)
        - Win rate consistency across periods
        - Return volatility
        """
        if sample_size == 0:
            return 0

        # Base confidence from sample size (log scale)
        size_factor = min(1.0, np.log10(sample_size + 1) / 2)

        # Consistency factor (are win rates similar across periods?)
        win_rates = [s["win_rate"] for s in stats.values() if "win_rate" in s]
        if win_rates:
            consistency = 1 - np.std(win_rates)
        else:
            consistency = 0.5

        # Signal strength (distance from 50% win rate)
        avg_win_rate = np.mean(win_rates) if win_rates else 0.5
        signal_strength = abs(avg_win_rate - 0.5) * 2

        # Combine factors
        confidence = (
            size_factor * 0.4 +
            consistency * 0.3 +
            signal_strength * 0.3
        )

        return min(1.0, max(0.0, confidence))

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            "total_matches": 0,
            "win_rate_1d": 0,
            "win_rate_5d": 0,
            "win_rate_20d": 0,
            "avg_return_1d": 0,
            "avg_return_5d": 0,
            "avg_return_20d": 0,
            "median_return_5d": 0,
            "confidence": 0,
        }

    def generate_report(
        self,
        query_info: Dict[str, Any],
        similar_patterns: List[Any],
        analysis: Dict[str, Any],
    ) -> str:
        """
        Generate a human-readable analysis report.

        Args:
            query_info: Information about the query pattern
            similar_patterns: List of similar patterns
            analysis: Analysis statistics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("K-LINE PATTERN SIMILARITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Query information
        report.append(f"Query: {query_info.get('symbol', 'N/A')} ({query_info.get('market', 'N/A')})")
        report.append(f"Period: {query_info.get('start_date', 'N/A')} to {query_info.get('end_date', 'N/A')}")
        report.append(f"Window Size: {query_info.get('window_size', 'N/A')} days")
        report.append("")

        # Summary statistics
        report.append("-" * 40)
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Similar Patterns Found: {analysis.get('total_matches', 0)}")
        report.append(f"Confidence Score: {analysis.get('confidence', 0):.1%}")
        report.append("")

        # Win rates
        report.append("Win Rates (price increase):")
        report.append(f"  T+1 day:   {analysis.get('win_rate_1d', 0):.1%}")
        report.append(f"  T+5 days:  {analysis.get('win_rate_5d', 0):.1%}")
        report.append(f"  T+20 days: {analysis.get('win_rate_20d', 0):.1%}")
        report.append("")

        # Average returns
        report.append("Average Returns:")
        report.append(f"  T+1 day:   {analysis.get('avg_return_1d', 0):+.2%}")
        report.append(f"  T+5 days:  {analysis.get('avg_return_5d', 0):+.2%}")
        report.append(f"  T+20 days: {analysis.get('avg_return_20d', 0):+.2%}")
        report.append("")

        # Interpretation
        report.append("-" * 40)
        report.append("INTERPRETATION")
        report.append("-" * 40)

        win_rate_5d = analysis.get('win_rate_5d', 0.5)
        if win_rate_5d > 0.65:
            interpretation = "Strong bullish signal. Historical patterns show high probability of price increase."
        elif win_rate_5d > 0.55:
            interpretation = "Moderately bullish. Consider combining with other indicators."
        elif win_rate_5d > 0.45:
            interpretation = "Neutral signal. No clear directional bias from historical patterns."
        elif win_rate_5d > 0.35:
            interpretation = "Moderately bearish. Historical patterns suggest downside risk."
        else:
            interpretation = "Strong bearish signal. Historical patterns show high probability of price decrease."

        report.append(interpretation)
        report.append("")

        # Top similar patterns
        if similar_patterns:
            report.append("-" * 40)
            report.append("TOP SIMILAR PATTERNS")
            report.append("-" * 40)
            for i, pattern in enumerate(similar_patterns[:5], 1):
                if hasattr(pattern, "symbol"):
                    symbol = pattern.symbol
                    score = pattern.similarity_score
                    start = pattern.start_date
                else:
                    symbol = pattern.get("symbol", "N/A")
                    score = pattern.get("similarity_score", 0)
                    start = pattern.get("start_date", "N/A")

                report.append(f"  {i}. {symbol} ({start}) - Similarity: {score:.1%}")

        report.append("")
        report.append("=" * 60)
        report.append("Note: Past performance does not guarantee future results.")
        report.append("=" * 60)

        return "\n".join(report)
