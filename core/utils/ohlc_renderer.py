"""
OHLC Bar chart renderer - Xiu et al. (2021) official implementation style.

Reference: "(Re-)Imag(in)ing Price Trends" (Xiu et al., 2021)

Image format:
- Black background (0)
- White OHLC bars (255)
- Bar width: 3 pixels
- Line width (high-low): 1 pixel
- Open tick: left horizontal line
- Close tick: right horizontal line
- Volume bars at bottom (optional)

Image sizes:
- 5 days: 15 x 32 pixels
- 20 days: 60 x 64 pixels
- 60 days: 180 x 96 pixels
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import Tuple, Union, Optional
from pathlib import Path


class OHLCRenderer:
    """
    Render OHLC Bar charts as sparse binary images (Xiu et al. style).
    """

    # Official image dimensions from paper
    BAR_WIDTH = 3
    LINE_WIDTH = 1
    IMAGE_WIDTH = {5: 15, 20: 60, 60: 96}  # bar_width * n_bars
    IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}
    VOLUME_CHART_GAP = 1
    
    # Colors: black background, white chart
    BACKGROUND_COLOR = 0
    CHART_COLOR = 255

    def __init__(self, n_bars: int = 20, include_volume: bool = False):
        """
        Initialize OHLC renderer.

        Args:
            n_bars: Number of OHLC bars (5, 20, or 60)
            include_volume: Whether to include volume bars at bottom
        """
        if n_bars not in [5, 20, 60]:
            raise ValueError(f"n_bars must be 5, 20, or 60, got {n_bars}")
        
        self.n_bars = n_bars
        self.include_volume = include_volume
        
        self.width = self.IMAGE_WIDTH[n_bars]
        self.height = self.IMAGE_HEIGHT[n_bars]
        
        # Calculate volume height if needed (Xiu et al. style)
        if include_volume:
            self.volume_height = int(self.height / 5)  # 20% for volume
            self.ohlc_height = self.height - self.volume_height - self.VOLUME_CHART_GAP
        else:
            self.volume_height = 0
            self.ohlc_height = self.height
        
        # Calculate bar centers (X coordinates)
        first_center = (self.BAR_WIDTH - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + self.BAR_WIDTH * n_bars,
            self.BAR_WIDTH,
            dtype=float,
        )

    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize prices so first close = 1.0 (Xiu et al. method).
        """
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        # Check if columns exist (case insensitive)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in price_cols:
                col_map[col] = col_lower
        df = df.rename(columns=col_map)
        
        # Handle timezone-aware index
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Normalize: first close = 1.0
        first_close = df['close'].iloc[0]
        if first_close == 0 or pd.isna(first_close):
            raise ValueError("First close price is zero or NaN")
        
        for col in price_cols:
            df[col] = df[col] / first_close
        
        return df

    def _price_to_y(self, price: float, min_p: float, max_p: float) -> int:
        """Convert price to Y coordinate (pixels)."""
        if max_p == min_p:
            return self.ohlc_height // 2
        
        pixels_per_unit = (self.ohlc_height - 1.0) / (max_p - min_p)
        y = int(np.round((price - min_p) * pixels_per_unit))
        
        # Clamp to valid range
        return max(0, min(self.ohlc_height - 1, y))

    def render(self, df: Union[pd.DataFrame, np.ndarray]) -> Image.Image:
        """
        Render OHLC data as Xiu et al. style bar chart.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            PIL Image (grayscale)
        """
        # Normalize column names and prices
        if isinstance(df, pd.DataFrame):
            df = self._normalize_prices(df)
        else:
            raise ValueError("Input must be a DataFrame")
        
        if len(df) != self.n_bars:
            raise ValueError(f"DataFrame must have exactly {self.n_bars} rows, got {len(df)}")
        
        # Get min/max for Y-axis scaling
        price_cols = ['open', 'high', 'low', 'close']
        min_p = df[price_cols].min().min()
        max_p = df[price_cols].max().max()
        
        if min_p == max_p or pd.isna(min_p) or pd.isna(max_p):
            raise ValueError("Invalid price range")
        
        # Create OHLC image
        ohlc_img = self._draw_ohlc(df, min_p, max_p)
        
        # Add volume if requested
        if self.include_volume and 'volume' in df.columns:
            volume_img = self._draw_volume(df)
            
            # Combine images
            full_img = Image.new(
                "L",
                (self.width, self.ohlc_height + self.volume_height + self.VOLUME_CHART_GAP),
                self.BACKGROUND_COLOR
            )
            full_img.paste(ohlc_img, (0, self.volume_height + self.VOLUME_CHART_GAP))
            full_img.paste(volume_img, (0, 0))
            
            # Flip vertically (Xiu et al. style)
            full_img = full_img.transpose(Image.FLIP_TOP_BOTTOM)
            return full_img
        else:
            # Flip vertically (Xiu et al. style)
            ohlc_img = ohlc_img.transpose(Image.FLIP_TOP_BOTTOM)
            return ohlc_img

    def _draw_ohlc(self, df: pd.DataFrame, min_p: float, max_p: float) -> Image.Image:
        """Draw the OHLC bars."""
        img = Image.new("L", (self.width, self.ohlc_height), self.BACKGROUND_COLOR)
        pixels = img.load()
        
        for day in range(self.n_bars):
            row = df.iloc[day]
            
            high_p = row['high']
            low_p = row['low']
            open_p = row['open']
            close_p = row['close']
            
            if pd.isna(high_p) or pd.isna(low_p):
                continue
            
            # Calculate X positions
            left = int(np.ceil(self.centers[day] - self.BAR_WIDTH // 2))
            right = int(np.floor(self.centers[day] + self.BAR_WIDTH // 2))
            line_left = int(np.ceil(self.centers[day] - self.LINE_WIDTH // 2))
            line_right = int(np.floor(self.centers[day] + self.LINE_WIDTH // 2))
            
            # Calculate Y positions
            line_bottom = self._price_to_y(low_p, min_p, max_p)
            line_top = self._price_to_y(high_p, min_p, max_p)
            
            # Draw vertical line (high to low) - thick bar
            for i in range(line_left, line_right + 1):
                for j in range(line_bottom, line_top + 1):
                    pixels[i, j] = self.CHART_COLOR
            
            # Draw open tick (left horizontal line)
            if not pd.isna(open_p):
                open_y = self._price_to_y(open_p, min_p, max_p)
                for i in range(left, int(self.centers[day]) + 1):
                    pixels[i, open_y] = self.CHART_COLOR
            
            # Draw close tick (right horizontal line)
            if not pd.isna(close_p):
                close_y = self._price_to_y(close_p, min_p, max_p)
                for i in range(int(self.centers[day]) + 1, right + 1):
                    pixels[i, close_y] = self.CHART_COLOR
        
        return img

    def _draw_volume(self, df: pd.DataFrame) -> Image.Image:
        """Draw volume bars at the top."""
        img = Image.new("L", (self.width, self.volume_height), self.BACKGROUND_COLOR)
        pixels = img.load()
        
        vol_col = 'volume' if 'volume' in df.columns else 'Volume'
        volumes = df[vol_col].values
        max_volume = np.max(np.abs(volumes))
        
        if np.isnan(max_volume) or max_volume == 0:
            return img
        
        pixels_per_volume = self.volume_height / max_volume
        
        for day in range(self.n_bars):
            if pd.isna(volumes[day]):
                continue
            
            vol_h = int(np.round(np.abs(volumes[day]) * pixels_per_volume))
            
            # Draw vertical line for volume
            x = int(self.centers[day])
            for y in range(vol_h):
                pixels[x, y] = self.CHART_COLOR
        
        return img

    def render_to_numpy(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Render and convert to numpy array."""
        img = self.render(df)
        return np.array(img)

    def render_batch(
        self,
        windows: list,
        output_dir: Union[str, Path],
        naming_pattern: str = "{idx:06d}.png"
    ) -> list:
        """Render multiple windows and save to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for idx, window in enumerate(windows):
            img = self.render(window)
            filename = naming_pattern.format(idx=idx)
            filepath = output_dir / filename
            img.save(filepath)
            saved_paths.append(filepath)

        return saved_paths


class ImageLibrary:
    """Manage rendered OHLC image library."""

    def __init__(self, library_dir: Union[str, Path], n_bars: int = 20, include_volume: bool = False):
        """
        Initialize image library.

        Args:
            library_dir: Root directory for the image library
            n_bars: Number of bars per image (5, 20, or 60)
            include_volume: Whether to include volume bars
        """
        self.library_dir = Path(library_dir)
        self.n_bars = n_bars
        self.include_volume = include_volume
        self.renderer = OHLCRenderer(n_bars=n_bars, include_volume=include_volume)

        # Create directory structure
        self.raw_dir = self.library_dir / "raw"
        self.processed_dir = self.library_dir / "processed"
        self.metadata_file = self.library_dir / "metadata.csv"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def add_stock_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        stride: int = 1,
    ) -> dict:
        """
        Add stock data to library by rendering windows.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            stride: Step size between windows

        Returns:
            Dictionary with metadata
        """
        symbol_dir = self.processed_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        # Normalize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                col_map[col] = col_lower
        df = df.rename(columns=col_map)
        
        # Handle timezone-aware index
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        # Generate windows
        n = len(df)
        metadata = []

        for i in range(0, n - self.n_bars + 1, stride):
            window = df.iloc[i:i + self.n_bars].copy()

            try:
                # Render image
                img = self.renderer.render(window)

                # Save image
                start_date = window.index[0]
                end_date = window.index[-1]
                
                if hasattr(start_date, 'strftime'):
                    start_str = start_date.strftime("%Y%m%d")
                    end_str = end_date.strftime("%Y%m%d")
                else:
                    start_str = str(start_date)
                    end_str = str(end_date)
                    
                filename = f"{symbol}_{start_str}_{end_str}_{i:06d}.png"
                filepath = symbol_dir / filename
                img.save(filepath)

                # Record metadata
                open_price = window['open'].iloc[0]
                close_price = window['close'].iloc[-1]
                ret = close_price / open_price - 1
                
                metadata.append({
                    'symbol': symbol,
                    'filename': filename,
                    'filepath': str(filepath.relative_to(self.library_dir)),
                    'window_start': i,
                    'window_end': i + self.n_bars,
                    'start_date': start_date,
                    'end_date': end_date,
                    'open_price': open_price,
                    'close_price': close_price,
                    'return': ret,
                })
            except Exception as e:
                print(f"  Error rendering window {i}: {e}")

        # Update metadata
        self._update_metadata(metadata)

        return {
            'symbol': symbol,
            'num_images': len(metadata),
            'directory': str(symbol_dir),
        }

    def _update_metadata(self, new_entries: list):
        """Update metadata CSV file."""
        if not new_entries:
            return
            
        if self.metadata_file.exists():
            existing = pd.read_csv(self.metadata_file)
            updated = pd.concat([existing, pd.DataFrame(new_entries)], ignore_index=True)
        else:
            updated = pd.DataFrame(new_entries)

        updated.to_csv(self.metadata_file, index=False)

    def get_metadata(self) -> pd.DataFrame:
        """Get full metadata DataFrame."""
        if self.metadata_file.exists():
            return pd.read_csv(self.metadata_file)
        return pd.DataFrame()

    def get_symbol_images(self, symbol: str) -> list:
        """Get all image paths for a symbol."""
        symbol_dir = self.processed_dir / symbol
        if not symbol_dir.exists():
            return []
        return sorted(symbol_dir.glob("*.png"))


if __name__ == "__main__":
    # Test rendering
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from core.tradingview_fetcher import TradingViewFetcher

    print("Testing Xiu et al. OHLC Renderer...")

    # Fetch test data
    fetcher = TradingViewFetcher()
    df = fetcher.fetch_ohlcv("AAPL", n_bars=20)

    if df is not None:
        print(f"Fetched {len(df)} bars")
        print(df.head())

        # Render as OHLC bars - 20 days, Xiu et al. style
        renderer = OHLCRenderer(n_bars=20, include_volume=True)
        img = renderer.render(df)

        # Save test image
        test_path = Path("test_ohlc_xiu_official.png")
        img.save(test_path)
        print(f"Saved test image to {test_path}")
        print(f"Image size: {img.size}")
        
        # Also save as numpy to verify
        img_np = np.array(img)
        print(f"Numpy shape: {img_np.shape}")
        print(f"Unique values: {np.unique(img_np)}")
    else:
        print("Failed to fetch data")
