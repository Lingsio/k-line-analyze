"""
Dynamic Time Warping (DTW) matcher for K-line pattern refinement.

DTW handles patterns that occur at different speeds/scales,
making it ideal for refining FAISS candidates.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.interpolate import interp1d


class DTWMatcher:
    """
    DTW-based pattern matcher for time series refinement.

    Used to re-rank FAISS candidates based on actual time series
    similarity, handling temporal distortions.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        use_fast_dtw: bool = True,
    ):
        """
        Initialize DTW matcher.

        Args:
            window: Sakoe-Chiba band width (None for full DTW)
            use_fast_dtw: Use FastDTW approximation for speed
        """
        self.window = window
        self.use_fast_dtw = use_fast_dtw

    def compute_dtw_distance(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute DTW distance between two time series.

        Args:
            series1: First time series
            series2: Second time series

        Returns:
            Tuple of (distance, warping_path)
        """
        series1 = np.asarray(series1, dtype=np.float64)
        series2 = np.asarray(series2, dtype=np.float64)

        # Normalize both series
        series1 = self._normalize(series1)
        series2 = self._normalize(series2)

        # Resample to same length if very different
        if abs(len(series1) - len(series2)) > len(series1) * 0.3:
            target_len = max(len(series1), len(series2))
            series1 = self._resample(series1, target_len)
            series2 = self._resample(series2, target_len)

        try:
            if self.use_fast_dtw:
                from dtw import dtw as dtw_lib
                alignment = dtw_lib(
                    series1, series2,
                    keep_internals=True,
                    step_pattern="symmetric2",
                    window_type="sakoechiba" if self.window else None,
                    window_args={"window_size": self.window} if self.window else {},
                )
                distance = alignment.normalizedDistance
                path = np.array(list(zip(alignment.index1, alignment.index2)))
            else:
                distance, path = self._compute_dtw_basic(series1, series2)
        except ImportError:
            # Fallback to basic implementation
            distance, path = self._compute_dtw_basic(series1, series2)

        return distance, path

    def _compute_dtw_basic(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Basic DTW implementation without external libraries.

        Args:
            series1: First series
            series2: Second series

        Returns:
            Tuple of (distance, path)
        """
        n, m = len(series1), len(series2)

        # Cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        # Fill the matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Apply window constraint if specified
                if self.window is not None and abs(i - j) > self.window:
                    continue

                cost = (series1[i - 1] - series2[j - 1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        # Compute normalized distance
        distance = np.sqrt(dtw_matrix[n, m]) / (n + m)

        # Backtrack to find optimal path
        path = self._backtrack(dtw_matrix, n, m)

        return distance, path

    def _backtrack(self, dtw_matrix: np.ndarray, n: int, m: int) -> np.ndarray:
        """Backtrack to find optimal warping path."""
        path = [(n - 1, m - 1)]
        i, j = n, m

        while i > 1 or j > 1:
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                candidates = [
                    (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
                    (dtw_matrix[i - 1, j], i - 1, j),
                    (dtw_matrix[i, j - 1], i, j - 1),
                ]
                _, i, j = min(candidates)

            path.append((i - 1, j - 1))

        path.reverse()
        return np.array(path)

    def _normalize(self, series: np.ndarray) -> np.ndarray:
        """Z-score normalize a series."""
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            return series - mean
        return (series - mean) / std

    def _resample(self, series: np.ndarray, target_len: int) -> np.ndarray:
        """Resample series to target length."""
        if len(series) == target_len:
            return series

        x_old = np.linspace(0, 1, len(series))
        x_new = np.linspace(0, 1, target_len)
        f = interp1d(x_old, series, kind="linear")
        return f(x_new)

    def refine_matches(
        self,
        query_series: np.ndarray,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Refine candidate matches using DTW.

        Args:
            query_series: Query time series (e.g., normalized close prices)
            candidates: List of candidate matches from FAISS
            top_k: Number of top matches to return

        Returns:
            Re-ranked candidates with DTW distances
        """
        if not candidates:
            return []

        refined = []

        for candidate in candidates:
            # Get the candidate's time series
            candidate_series = candidate.get("series")

            if candidate_series is None:
                # If series not available, use original similarity score
                refined.append({
                    **candidate,
                    "dtw_distance": None,
                    "combined_score": candidate.get("similarity_score", 0),
                })
                continue

            # Compute DTW distance
            dtw_dist, _ = self.compute_dtw_distance(query_series, candidate_series)

            # Combine FAISS similarity with DTW distance
            # Lower DTW = better match, higher similarity = better match
            similarity = candidate.get("similarity_score", 0.5)
            combined_score = similarity * np.exp(-dtw_dist)

            refined.append({
                **candidate,
                "dtw_distance": float(dtw_dist),
                "combined_score": float(combined_score),
            })

        # Sort by combined score (descending)
        refined.sort(key=lambda x: x["combined_score"], reverse=True)

        return refined[:top_k]

    def align_series(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two series using DTW for visualization.

        Args:
            series1: First series
            series2: Second series

        Returns:
            Tuple of aligned series
        """
        _, path = self.compute_dtw_distance(series1, series2)

        # Create aligned series
        aligned1 = series1[path[:, 0]]
        aligned2 = series2[path[:, 1]]

        return aligned1, aligned2


class ShapeDTW(DTWMatcher):
    """
    Shape-based DTW that focuses on local shape features.

    More robust to amplitude differences and better for pattern matching.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        descriptor_type: str = "slope",
    ):
        """
        Initialize Shape DTW.

        Args:
            window: DTW window size
            descriptor_type: 'slope', 'derivative', or 'paa'
        """
        super().__init__(window=window, use_fast_dtw=False)
        self.descriptor_type = descriptor_type

    def _extract_shape_descriptor(self, series: np.ndarray) -> np.ndarray:
        """
        Extract shape descriptor from time series.

        Args:
            series: Input time series

        Returns:
            Shape descriptor array
        """
        if self.descriptor_type == "slope":
            # Piecewise slope
            return np.diff(series)

        elif self.descriptor_type == "derivative":
            # Smooth derivative
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(series, sigma=2)
            return np.gradient(smoothed)

        elif self.descriptor_type == "paa":
            # Piecewise Aggregate Approximation
            n_segments = min(20, len(series) // 2)
            segment_size = len(series) // n_segments
            paa = np.array([
                np.mean(series[i * segment_size:(i + 1) * segment_size])
                for i in range(n_segments)
            ])
            return paa

        return series

    def compute_dtw_distance(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute Shape DTW distance.

        Args:
            series1: First series
            series2: Second series

        Returns:
            Tuple of (distance, path)
        """
        # Extract shape descriptors
        desc1 = self._extract_shape_descriptor(series1)
        desc2 = self._extract_shape_descriptor(series2)

        # Compute DTW on shape descriptors
        return super().compute_dtw_distance(desc1, desc2)
