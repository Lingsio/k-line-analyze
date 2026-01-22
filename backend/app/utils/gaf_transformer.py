"""
Gramian Angular Field (GAF) transformer for time series to image conversion.

GAF encodes temporal correlations into a 2D image format, making it suitable
for CNN-based pattern recognition.

References:
- Wang & Oates (2015): "Imaging Time-Series to Improve Classification and Imputation"
"""

import numpy as np
from typing import Literal


class GAFTransformer:
    """Transform time series data into Gramian Angular Field images."""

    def __init__(
        self,
        image_size: int = 128,
        method: Literal["gasf", "gadf"] = "gasf",
    ):
        """
        Initialize GAF transformer.

        Args:
            image_size: Output image size (square)
            method: 'gasf' (Gramian Angular Summation Field) or
                   'gadf' (Gramian Angular Difference Field)
        """
        self.image_size = image_size
        self.method = method

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Transform a 1D time series to a 2D GAF image.

        Args:
            series: 1D numpy array of time series values

        Returns:
            2D numpy array (GAF image)
        """
        # Step 1: Rescale to [-1, 1]
        series = np.array(series, dtype=np.float64)
        min_val = np.min(series)
        max_val = np.max(series)

        if max_val - min_val == 0:
            scaled = np.zeros_like(series)
        else:
            scaled = (2 * series - max_val - min_val) / (max_val - min_val)

        # Clip to ensure values are in [-1, 1]
        scaled = np.clip(scaled, -1, 1)

        # Step 2: Resample to image_size if needed
        if len(scaled) != self.image_size:
            scaled = self._resample(scaled, self.image_size)

        # Step 3: Calculate angular values (polar encoding)
        # φ = arccos(x), where x is the scaled value
        phi = np.arccos(scaled)

        # Step 4: Compute Gramian Angular Field
        if self.method == "gasf":
            # GASF: cos(φ_i + φ_j)
            gaf = np.cos(phi[:, np.newaxis] + phi[np.newaxis, :])
        else:
            # GADF: sin(φ_i - φ_j)
            gaf = np.sin(phi[:, np.newaxis] - phi[np.newaxis, :])

        return gaf

    def _resample(self, series: np.ndarray, target_len: int) -> np.ndarray:
        """Resample series to target length using linear interpolation."""
        from scipy.interpolate import interp1d

        x_old = np.linspace(0, 1, len(series))
        x_new = np.linspace(0, 1, target_len)

        f = interp1d(x_old, series, kind="linear")
        return f(x_new)

    def transform_batch(self, series_list: list) -> np.ndarray:
        """
        Transform multiple time series to GAF images.

        Args:
            series_list: List of 1D numpy arrays

        Returns:
            3D numpy array (batch_size x image_size x image_size)
        """
        batch = np.zeros((len(series_list), self.image_size, self.image_size))

        for i, series in enumerate(series_list):
            batch[i] = self.transform(series)

        return batch

    def inverse_transform(self, gaf: np.ndarray) -> np.ndarray:
        """
        Approximate inverse transform from GAF to time series.

        Note: This is an approximation since GAF is not perfectly invertible.

        Args:
            gaf: 2D GAF image

        Returns:
            1D approximate time series
        """
        # Extract diagonal elements (self-correlation)
        if self.method == "gasf":
            # For GASF: diagonal = cos(2φ)
            diagonal = np.diag(gaf)
            # φ = arccos(diagonal) / 2
            phi = np.arccos(np.clip(diagonal, -1, 1)) / 2
        else:
            # For GADF: diagonal is always 0, use first row instead
            first_row = gaf[0]
            phi = np.arcsin(np.clip(first_row, -1, 1))

        # Convert back to [-1, 1] scale
        series = np.cos(phi)

        return series


class MTFTransformer:
    """
    Markov Transition Field (MTF) transformer.

    Alternative to GAF that captures transition probabilities between
    quantile bins across time steps.
    """

    def __init__(self, image_size: int = 128, n_bins: int = 8):
        """
        Initialize MTF transformer.

        Args:
            image_size: Output image size
            n_bins: Number of quantile bins
        """
        self.image_size = image_size
        self.n_bins = n_bins

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Transform time series to Markov Transition Field.

        Args:
            series: 1D numpy array

        Returns:
            2D MTF image
        """
        # Resample if needed
        if len(series) != self.image_size:
            from scipy.interpolate import interp1d

            x_old = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, self.image_size)
            f = interp1d(x_old, series, kind="linear")
            series = f(x_new)

        # Quantize to bins
        bins = np.percentile(series, np.linspace(0, 100, self.n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        quantized = np.digitize(series, bins) - 1
        quantized = np.clip(quantized, 0, self.n_bins - 1)

        # Compute transition matrix
        transition_matrix = np.zeros((self.n_bins, self.n_bins))
        for i in range(len(quantized) - 1):
            transition_matrix[quantized[i], quantized[i + 1]] += 1

        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums

        # Create MTF image
        mtf = np.zeros((self.image_size, self.image_size))
        for i in range(self.image_size):
            for j in range(self.image_size):
                mtf[i, j] = transition_matrix[quantized[i], quantized[j]]

        return mtf
