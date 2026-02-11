"""
K-Line (Candlestick) chart renderer for image generation.

Generates clean, noise-free candlestick images suitable for CNN input.
No axes, gridlines, or labels - pure pattern representation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


class KLineRenderer:
    """Render K-line candlestick charts as images."""

    def __init__(
        self,
        image_size: int = 128,
        bullish_color: Tuple[int, int, int] = (38, 166, 154),  # Green
        bearish_color: Tuple[int, int, int] = (239, 83, 80),   # Red
        wick_color: Tuple[int, int, int] = (200, 200, 200),    # Gray
        background_color: Tuple[int, int, int] = (13, 17, 23), # Dark
        style: str = "default",
    ):
        """
        Initialize K-line renderer.

        Args:
            image_size: Output image size (square)
            bullish_color: RGB color for bullish (up) candles
            bearish_color: RGB color for bearish (down) candles
            wick_color: RGB color for candle wicks
            background_color: RGB background color
            style: 'default' or 'paper' (Jiang et al. 2023 style)
        """
        self.image_size = image_size
        self.style = style
        
        if style == "paper":
            self.bullish_color = (255, 255, 255) # White
            self.bearish_color = (128, 128, 128) # Gray
            self.wick_color = (180, 180, 180)
            self.background_color = (0, 0, 0)
        else:
            self.bullish_color = bullish_color
            self.bearish_color = bearish_color
            self.wick_color = wick_color
            self.background_color = background_color

    def render(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        include_volume: bool = True,
        volume_height_ratio: float = 0.2,
    ) -> np.ndarray:
        """
        Render K-line data as an RGB image.

        Args:
            df: DataFrame with normalized OHLCV data (values in [0, 1])
            include_volume: Whether to include volume bars
            volume_height_ratio: Ratio of image height for volume bars

        Returns:
            RGB numpy array (image_size x image_size x 3)
        """
        # Create blank image
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image[:, :] = self.background_color

        n_candles = len(df)
        if n_candles == 0:
            return image

        # Calculate dimensions
        if include_volume:
            price_height = int(self.image_size * (1 - volume_height_ratio))
            volume_height = self.image_size - price_height
        else:
            price_height = self.image_size
            volume_height = 0

        candle_width = max(1, self.image_size // n_candles)
        candle_spacing = max(1, candle_width // 4)
        candle_body_width = max(1, candle_width - candle_spacing)

        # Get OHLC values
        if isinstance(df, np.ndarray):
            # Numpy input: assume [open, high, low, close, volume]
            opens = df[:, 0]
            highs = df[:, 1]
            lows = df[:, 2]
            closes = df[:, 3]
        else:
            # DataFrame input
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values

        # Ensure values are in [0, 1]
        all_prices = np.concatenate([opens, highs, lows, closes])
        price_min = np.min(all_prices)
        price_max = np.max(all_prices)

        if price_max - price_min == 0:
            price_max = price_min + 1

        # Normalize if not already
        def normalize_price(p):
            return (p - price_min) / (price_max - price_min)

        # Draw each candle
        for i in range(n_candles):
            x_center = int((i + 0.5) * self.image_size / n_candles)
            x_left = x_center - candle_body_width // 2
            x_right = x_left + candle_body_width

            # Ensure x coordinates are within bounds
            x_left = max(0, min(x_left, self.image_size - 1))
            x_right = max(0, min(x_right, self.image_size - 1))

            # Normalize prices
            o = normalize_price(opens[i])
            h = normalize_price(highs[i])
            l = normalize_price(lows[i])
            c = normalize_price(closes[i])

            # Convert to pixel coordinates (invert y-axis)
            def price_to_y(p):
                return int((1 - p) * (price_height - 1))

            y_open = price_to_y(o)
            y_high = price_to_y(h)
            y_low = price_to_y(l)
            y_close = price_to_y(c)

            # Determine if bullish or bearish
            is_bullish = c >= o
            body_color = self.bullish_color if is_bullish else self.bearish_color

            # Draw wick (vertical line from high to low)
            wick_x = x_center
            if 0 <= wick_x < self.image_size:
                y_top = min(y_high, y_low)
                y_bottom = max(y_high, y_low)
                y_top = max(0, min(y_top, price_height - 1))
                y_bottom = max(0, min(y_bottom, price_height - 1))

                for y in range(y_top, y_bottom + 1):
                    image[y, wick_x] = self.wick_color

            # Draw body (rectangle from open to close)
            y_body_top = min(y_open, y_close)
            y_body_bottom = max(y_open, y_close)
            y_body_top = max(0, min(y_body_top, price_height - 1))
            y_body_bottom = max(0, min(y_body_bottom, price_height - 1))

            # Ensure at least 1 pixel height for body
            if y_body_top == y_body_bottom:
                y_body_bottom = min(y_body_top + 1, price_height - 1)

            for y in range(y_body_top, y_body_bottom + 1):
                for x in range(x_left, x_right + 1):
                    if 0 <= x < self.image_size and 0 <= y < price_height:
                        image[y, x] = body_color

        # Draw volume bars if requested
        has_volume = False
        if isinstance(df, np.ndarray):
            if df.shape[1] >= 5:
                volumes = df[:, 4]
                has_volume = True
        elif "volume" in df.columns:
            volumes = df["volume"].values
            has_volume = True

        if include_volume and has_volume:
            vol_max = np.max(volumes) if np.max(volumes) > 0 else 1

            for i in range(n_candles):
                x_center = int((i + 0.5) * self.image_size / n_candles)
                x_left = x_center - candle_body_width // 2
                x_right = x_left + candle_body_width

                x_left = max(0, min(x_left, self.image_size - 1))
                x_right = max(0, min(x_right, self.image_size - 1))

                vol_normalized = volumes[i] / vol_max
                vol_bar_height = int(vol_normalized * (volume_height - 2))
                vol_bar_height = max(1, vol_bar_height)

                y_vol_top = price_height + (volume_height - vol_bar_height)
                y_vol_bottom = self.image_size - 1

                # Use candle color for volume bars
                is_bullish = closes[i] >= opens[i]
                vol_color = self.bullish_color if is_bullish else self.bearish_color

                for y in range(y_vol_top, y_vol_bottom + 1):
                    for x in range(x_left, x_right + 1):
                        if 0 <= x < self.image_size and 0 <= y < self.image_size:
                            image[y, x] = vol_color

        return image

    def render_volume_only(self, df: pd.DataFrame) -> np.ndarray:
        """
        Render only volume bars as a grayscale image.

        Args:
            df: DataFrame with volume data

        Returns:
            Grayscale numpy array (image_size x image_size)
        """
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        if "volume" not in df.columns:
            return image

        n_bars = len(df)
        volumes = df["volume"].values
        vol_max = np.max(volumes) if np.max(volumes) > 0 else 1

        bar_width = max(1, self.image_size // n_bars)

        for i in range(n_bars):
            x_center = int((i + 0.5) * self.image_size / n_bars)
            x_left = x_center - bar_width // 2
            x_right = x_left + bar_width

            x_left = max(0, min(x_left, self.image_size - 1))
            x_right = max(0, min(x_right, self.image_size - 1))

            vol_normalized = volumes[i] / vol_max
            bar_height = int(vol_normalized * (self.image_size - 2))
            bar_height = max(1, bar_height)

            y_top = self.image_size - bar_height
            y_bottom = self.image_size - 1

            intensity = int(vol_normalized * 255)

            for y in range(y_top, y_bottom + 1):
                for x in range(x_left, x_right + 1):
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        image[y, x] = intensity

        return image

    def render_line_chart(self, df: pd.DataFrame, column: str = "close") -> np.ndarray:
        """
        Render a simple line chart (useful for overlay comparisons).

        Args:
            df: DataFrame with price data
            column: Column to plot

        Returns:
            Grayscale numpy array
        """
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        values = df[column].values
        n_points = len(values)

        if n_points < 2:
            return image

        # Normalize values
        v_min = np.min(values)
        v_max = np.max(values)
        if v_max - v_min == 0:
            v_max = v_min + 1

        normalized = (values - v_min) / (v_max - v_min)

        # Draw line connecting points
        for i in range(n_points - 1):
            x1 = int(i * self.image_size / n_points)
            x2 = int((i + 1) * self.image_size / n_points)
            y1 = int((1 - normalized[i]) * (self.image_size - 1))
            y2 = int((1 - normalized[i + 1]) * (self.image_size - 1))

            # Bresenham's line algorithm
            self._draw_line(image, x1, y1, x2, y2, 255)

        return image

    def _draw_line(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: int,
    ):
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < self.image_size and 0 <= y1 < self.image_size:
                image[y1, x1] = color

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
