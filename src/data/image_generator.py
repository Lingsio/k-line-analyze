import matplotlib.pyplot as plt
import mplfinance as mpf
import io
import numpy as np
from PIL import Image
import cv2
from typing import Literal, Optional


class ImageGenerator:
    """
    Enhanced K-line image generator with multiple normalization strategies
    and advanced visualization options.
    """
    def __init__(self, img_size=(128, 128), norm_method='robust', include_volume=True):
        """
        Args:
            img_size: Output image size (default increased to 128x128)
            norm_method: 'minmax', 'robust', 'percentile', 'zscore'
            include_volume: Whether to include volume bars in image
        """
        self.img_size = img_size
        self.norm_method = norm_method
        self.include_volume = include_volume

        # Style settings for chinese market (Red=Up, Green=Down)
        self.mc = mpf.make_marketcolors(up='red', down='green', edge='i', wick='i', volume='in', inherit=True)
        self.s = mpf.make_mpf_style(marketcolors=self.mc, gridstyle='', y_on_right=False)

    def dataframe_to_image(self, df_window):
        """
        Convert a DataFrame window (index=Date, cols=Open,High,Low,Close,Volume) to valid image tensor.
        Uses mplfinance but optimized.
        """
        # Create a buffer to save image
        buf = io.BytesIO()
        
        # Plotting
        # We turn off axes, labels to get pure data visualization
        fig, axes = mpf.plot(
            df_window,
            type='candle',
            returnfig=True,
            volume=True,
            style=self.s,
            title='',
            ylabel='',
            ylabel_lower='',
            figsize=(1, 1), # Small figure size
            tight_layout=True,
            axisoff=True,
            scale_padding=0
        )
        
        # Save to buffer
        fig.savefig(buf, format='png', dpi=self.img_size[0], bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Convert to numpy array
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = img.resize(self.img_size)
        img_np = np.array(img)
        
        # Normalize to 0-1 or standardize? usually 0-255 is fine for ToTensor later
        return img_np

    def _normalize_prices(self, open_p, high_p, low_p, close_p):
        """
        Apply advanced normalization strategies to price data.

        Returns normalized prices in range [0, 1] with better outlier handling.
        """
        all_prices = np.concatenate([open_p, high_p, low_p, close_p])

        if self.norm_method == 'minmax':
            # Standard MinMax
            min_p = np.min(all_prices)
            max_p = np.max(all_prices)

        elif self.norm_method == 'robust':
            # Robust scaling using IQR (handles outliers better)
            q1 = np.percentile(all_prices, 5)
            q3 = np.percentile(all_prices, 95)
            min_p = q1 - 0.5 * (q3 - q1)  # Extended range
            max_p = q3 + 0.5 * (q3 - q1)

        elif self.norm_method == 'percentile':
            # Percentile-based normalization (very robust to outliers)
            min_p = np.percentile(all_prices, 2)
            max_p = np.percentile(all_prices, 98)

        elif self.norm_method == 'zscore':
            # Z-score based (center at mean, scale by std)
            mean_p = np.mean(all_prices)
            std_p = np.std(all_prices) + 1e-8
            # Map to [0, 1] assuming most values within 3 std
            min_p = mean_p - 3 * std_p
            max_p = mean_p + 3 * std_p
        else:
            min_p = np.min(all_prices)
            max_p = np.max(all_prices)

        range_p = max_p - min_p
        if range_p < 1e-8:
            range_p = 1e-8

        # Normalize and clip to [0, 1]
        norm_open = np.clip((open_p - min_p) / range_p, 0, 1)
        norm_high = np.clip((high_p - min_p) / range_p, 0, 1)
        norm_low = np.clip((low_p - min_p) / range_p, 0, 1)
        norm_close = np.clip((close_p - min_p) / range_p, 0, 1)

        return norm_open, norm_high, norm_low, norm_close

    def fast_cv2_draw(self, open_p, high_p, low_p, close_p, volume, size=None, grayscale=False):
        """
        Enhanced K-line image generation with volume bars and better normalization.

        Args:
            open_p, high_p, low_p, close_p: OHLC price arrays
            volume: Volume array
            size: Image size tuple (H, W), uses self.img_size if None
            grayscale: If True, output 1-channel image
        """
        if size is None:
            size = self.img_size
        H, W = size
        channels = 1 if grayscale else 3
        img = np.zeros((H, W, channels), dtype=np.uint8)

        # Reserve space for volume if enabled
        if self.include_volume:
            price_height = int(H * 0.75)  # 75% for price chart
            vol_height = H - price_height - 2  # Rest for volume (with gap)
            vol_start_y = price_height + 2
        else:
            price_height = H
            vol_height = 0
            vol_start_y = H

        # Apply advanced normalization
        norm_open, norm_high, norm_low, norm_close = self._normalize_prices(
            open_p, high_p, low_p, close_p
        )

        def y_coord(normalized_price):
            # Map normalized [0,1] to pixel coordinates with padding
            padding = price_height * 0.05
            val = int(price_height - padding - normalized_price * (price_height - 2 * padding))
            return max(0, min(price_height - 1, val))

        num_candles = len(open_p)
        candle_w = max(1, (W - 4) // num_candles)  # Leave margin
        gap = max(1, candle_w // 4)  # Gap between candles
        actual_candle_w = max(1, candle_w - gap)

        for i in range(num_candles):
            x_center = int(2 + i * candle_w + candle_w / 2)

            y_h = y_coord(norm_high[i])
            y_l = y_coord(norm_low[i])
            y_o = y_coord(norm_open[i])
            y_c = y_coord(norm_close[i])

            is_up = close_p[i] >= open_p[i]

            if grayscale:
                # Enhanced grayscale: brightness encodes direction + magnitude
                magnitude = abs(close_p[i] - open_p[i]) / (high_p[i] - low_p[i] + 1e-8)
                base_intensity = 200 if is_up else 80
                intensity = int(base_intensity + magnitude * 55)
                color = (min(255, intensity),)
            else:
                # BGR: Red (Up), Green (Down)
                color = (0, 0, 255) if is_up else (0, 255, 0)

            # Draw Wick (shadow line)
            cv2.line(img, (x_center, y_h), (x_center, y_l), color, 1)

            # Draw Body
            top = min(y_o, y_c)
            bottom = max(y_o, y_c)

            x_left = max(0, x_center - actual_candle_w // 2)
            x_right = min(W - 1, x_center + actual_candle_w // 2)

            if x_right <= x_left:
                x_right = x_left + 1

            if top == bottom:
                # Doji
                cv2.line(img, (x_left, top), (x_right, top), color, 1)
            else:
                cv2.rectangle(img, (x_left, top), (x_right, bottom), color, -1)

        # Draw Volume bars
        if self.include_volume and vol_height > 2:
            max_vol = np.max(volume) + 1e-8
            norm_vol = volume / max_vol

            for i in range(num_candles):
                x_center = int(2 + i * candle_w + candle_w / 2)
                x_left = max(0, x_center - actual_candle_w // 2)
                x_right = min(W - 1, x_center + actual_candle_w // 2)

                bar_height = int(norm_vol[i] * (vol_height - 1))
                bar_top = vol_start_y + (vol_height - bar_height)
                bar_bottom = vol_start_y + vol_height - 1

                is_up = close_p[i] >= open_p[i]
                if grayscale:
                    vol_color = (180,) if is_up else (60,)
                else:
                    vol_color = (0, 0, 180) if is_up else (0, 180, 0)  # Dimmer colors

                if bar_height > 0 and x_right > x_left:
                    cv2.rectangle(img, (x_left, bar_top), (x_right, bar_bottom), vol_color, -1)

        return img

    def draw_with_augmentation(self, open_p, high_p, low_p, close_p, volume,
                                augment_type=None, size=None, grayscale=False):
        """
        Draw K-line with optional data augmentation.

        Args:
            augment_type: None, 'time_warp', 'cutout', 'noise', 'scale'
        """
        # Apply augmentation to data before drawing
        if augment_type == 'noise':
            noise_level = 0.02
            open_p = open_p * (1 + np.random.normal(0, noise_level, len(open_p)))
            high_p = high_p * (1 + np.random.normal(0, noise_level, len(high_p)))
            low_p = low_p * (1 + np.random.normal(0, noise_level, len(low_p)))
            close_p = close_p * (1 + np.random.normal(0, noise_level, len(close_p)))
            # Ensure OHLC consistency
            high_p = np.maximum.reduce([open_p, high_p, low_p, close_p])
            low_p = np.minimum.reduce([open_p, high_p, low_p, close_p])

        elif augment_type == 'scale':
            scale = np.random.uniform(0.95, 1.05)
            mean_p = (open_p.mean() + close_p.mean()) / 2
            open_p = (open_p - mean_p) * scale + mean_p
            high_p = (high_p - mean_p) * scale + mean_p
            low_p = (low_p - mean_p) * scale + mean_p
            close_p = (close_p - mean_p) * scale + mean_p

        elif augment_type == 'time_warp':
            # Simple time warping by stretching/compressing segments
            n = len(open_p)
            warp_points = np.sort(np.random.choice(range(1, n-1), size=min(3, n//5), replace=False))
            warp_points = np.concatenate([[0], warp_points, [n-1]])
            new_indices = []
            for i in range(len(warp_points)-1):
                start, end = warp_points[i], warp_points[i+1]
                segment_len = end - start
                warp_factor = np.random.uniform(0.8, 1.2)
                new_segment_len = max(1, int(segment_len * warp_factor))
                new_indices.extend(np.linspace(start, end-1, new_segment_len).astype(int))
            # Resample to original length
            if len(new_indices) != n:
                new_indices = np.linspace(0, len(new_indices)-1, n).astype(int)
                new_indices = np.clip(new_indices, 0, len(open_p)-1)
            new_indices = np.array(new_indices)
            new_indices = np.clip(new_indices, 0, len(open_p)-1)
            open_p = open_p[new_indices]
            high_p = high_p[new_indices]
            low_p = low_p[new_indices]
            close_p = close_p[new_indices]
            volume = volume[new_indices]

        # Draw the image
        img = self.fast_cv2_draw(open_p, high_p, low_p, close_p, volume, size, grayscale)

        # Post-drawing augmentation
        if augment_type == 'cutout':
            # Random rectangular cutout
            if size is None:
                size = self.img_size
            H, W = size
            cut_h = np.random.randint(H // 8, H // 4)
            cut_w = np.random.randint(W // 8, W // 4)
            cut_y = np.random.randint(0, H - cut_h)
            cut_x = np.random.randint(0, W - cut_w)
            img[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w] = 0

        return img

    # ============================================================
    # Advanced Image Processing Techniques
    # ============================================================

    def apply_clahe(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Enhances local contrast while preventing over-amplification of noise.

        Args:
            img: Input BGR image (H, W, 3)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization

        Returns:
            Enhanced BGR image
        """
        if len(img.shape) == 2 or img.shape[2] == 1:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(img.squeeze()).reshape(img.shape)

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def extract_edges(self, img, low_threshold=50, high_threshold=150):
        """
        Extract edge features using Canny edge detection.
        Useful for capturing K-line shape patterns.

        Args:
            img: Input image (H, W, C) or (H, W)
            low_threshold: Lower threshold for hysteresis
            high_threshold: Upper threshold for hysteresis

        Returns:
            Edge map (H, W) with values 0 or 255
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges

    def to_hsv(self, img):
        """
        Convert BGR image to HSV color space.
        Better for separating red/green colors in K-line charts.

        Returns:
            HSV image (H, W, 3)
        """
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def create_multi_channel(self, img, include_edge=True, include_hsv_h=True):
        """
        Create multi-channel image with additional feature channels.

        Channels:
        - 0-2: Original BGR
        - 3: Edge detection (if include_edge)
        - 4: HSV Hue channel (if include_hsv_h)

        Args:
            img: Input BGR image (H, W, 3)
            include_edge: Add edge detection channel
            include_hsv_h: Add HSV hue channel

        Returns:
            Multi-channel image (H, W, C) where C = 3 + extras
        """
        channels = [img]

        if include_edge:
            edges = self.extract_edges(img)
            channels.append(edges[:, :, np.newaxis])

        if include_hsv_h:
            hsv = self.to_hsv(img)
            hue = hsv[:, :, 0:1]  # Hue channel only
            channels.append(hue)

        return np.concatenate(channels, axis=2)

    def apply_gaussian_blur(self, img, kernel_size=3):
        """
        Apply Gaussian blur to reduce noise.

        Args:
            img: Input image
            kernel_size: Size of Gaussian kernel (odd number)

        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def letterbox_resize(self, img, target_size, fill_color=(0, 0, 0)):
        """
        Resize image while preserving aspect ratio using letterbox padding.

        Args:
            img: Input image (H, W, C)
            target_size: Target (H, W)
            fill_color: Color for padding

        Returns:
            Resized and padded image
        """
        h, w = img.shape[:2]
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas with fill color
        if len(img.shape) == 3:
            canvas = np.full((target_h, target_w, img.shape[2]), fill_color, dtype=np.uint8)
        else:
            canvas = np.full((target_h, target_w), fill_color[0], dtype=np.uint8)

        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    @staticmethod
    def mixup(img1, img2, alpha=0.4):
        """
        Mixup augmentation: blend two images.

        Args:
            img1, img2: Two images of same shape
            alpha: Beta distribution parameter

        Returns:
            Blended image, lambda value
        """
        lam = np.random.beta(alpha, alpha)
        mixed = (lam * img1.astype(np.float32) + (1 - lam) * img2.astype(np.float32))
        return mixed.astype(np.uint8), lam

    @staticmethod
    def cutmix(img1, img2, alpha=1.0):
        """
        CutMix augmentation: cut and paste rectangular region.

        Args:
            img1, img2: Two images of same shape
            alpha: Beta distribution parameter

        Returns:
            Mixed image, bounding box, lambda value
        """
        h, w = img1.shape[:2]
        lam = np.random.beta(alpha, alpha)

        # Get random bounding box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        mixed = img1.copy()
        mixed[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

        # Adjust lambda based on actual cut area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))

        return mixed, (x1, y1, x2, y2), lam

    def draw_ohlc_bars(self, open_p, high_p, low_p, close_p, volume,
                       size=None, grayscale=False, include_volume=True):
        """
        Draw OHLC bar chart (Xiu et al. style) - each day is 3 pixels wide.
        
        This format is more compact and was shown to be effective in:
        "(Re-)Imag(in)ing Price Trends" by Jiang, Kelly, Xiu (2021)
        
        Args:
            open_p, high_p, low_p, close_p: OHLC price arrays
            volume: Volume array
            size: Image size tuple (H, W), default 256x256
            grayscale: If True, output single channel
            include_volume: Whether to include volume bars at bottom
            
        Returns:
            OHLC bar chart image as numpy array
        """
        if size is None:
            size = self.img_size
        H, W = size
        
        # Create black background (sparse representation as in Xiu et al.)
        if grayscale:
            img = np.zeros((H, W), dtype=np.uint8)
        else:
            img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Reserve space for volume
        if include_volume and len(volume) > 0:
            price_height = int(H * 0.8)  # 80% for price
            vol_height = H - price_height - 2
            vol_start_y = price_height + 2
        else:
            price_height = H
            vol_height = 0
            vol_start_y = H
        
        # Normalization: Scale so max-min price range spans the height
        # This is the key insight from Xiu et al. - all stocks on same scale
        all_prices = np.concatenate([open_p, high_p, low_p, close_p])
        min_price = np.min(all_prices)
        max_price = np.max(all_prices)
        price_range = max_price - min_price
        
        if price_range < 1e-8:
            price_range = 1e-8
        
        # Padding to avoid edge effects
        padding_y = int(price_height * 0.05)
        effective_height = price_height - 2 * padding_y
        
        def price_to_y(price):
            """Map price to y-coordinate (inverted, so high price at top)"""
            normalized = (price - min_price) / price_range
            y = int(padding_y + (1 - normalized) * effective_height)
            return max(padding_y, min(price_height - padding_y - 1, y))
        
        # Each candle occupies 3 pixels width (as in Xiu et al.)
        num_bars = len(open_p)
        bar_width = 3
        total_width = num_bars * bar_width
        
        # Center the chart if smaller than image width
        x_offset = max(0, (W - total_width) // 2)
        
        for i in range(num_bars):
            x_base = x_offset + i * bar_width
            if x_base + 2 >= W:
                break
            
            y_open = price_to_y(open_p[i])
            y_high = price_to_y(high_p[i])
            y_low = price_to_y(low_p[i])
            y_close = price_to_y(close_p[i])
            
            # Color: White for visible elements on black background (Xiu et al. style)
            # Or traditional Red/Green for up/down
            is_up = close_p[i] >= open_p[i]
            
            if grayscale:
                color = 255  # White
            else:
                # Use white for all as in Xiu et al., or red/green
                # White on black is more sparse (better for CNN)
                color = (255, 255, 255)  # White
                # Alternative: traditional colors
                # color = (0, 0, 255) if is_up else (0, 255, 0)  # Red/Green
            
            # Draw high-low line (vertical bar) - center pixel (x_base + 1)
            cv2.line(img, (x_base + 1, y_high), (x_base + 1, y_low), color, 1)
            
            # Draw open tick (left pixel)
            cv2.line(img, (x_base, y_open), (x_base + 1, y_open), color, 1)
            
            # Draw close tick (right pixel)
            cv2.line(img, (x_base + 1, y_close), (x_base + 2, y_close), color, 1)
        
        # Draw volume bars at bottom
        if include_volume and vol_height > 2 and len(volume) > 0:
            max_vol = np.max(volume) + 1e-8
            
            for i in range(num_bars):
                x_base = x_offset + i * bar_width
                if x_base + 2 >= W:
                    break
                
                vol_normalized = volume[i] / max_vol
                bar_h = int(vol_normalized * (vol_height - 1))
                
                if bar_h > 0:
                    y_top = vol_start_y + (vol_height - bar_h)
                    y_bottom = vol_start_y + vol_height - 1
                    
                    if grayscale:
                        vol_color = 200
                    else:
                        vol_color = (200, 200, 200)  # Light gray
                    
                    cv2.rectangle(img, (x_base, y_top), (x_base + 2, y_bottom), vol_color, -1)
        
        return img

    # ============================================================
    # Sparse OHLC Bar Chart (compact grayscale representation)
    # ============================================================

    # Fixed image sizes per window size
    SPARSE_IMAGE_SIZES = {
        5:  (32, 15),
        20: (64, 60),
        60: (96, 180),
    }

    def draw_sparse_ohlc(self, open_p, high_p, low_p, close_p, volume,
                          window_size=20):
        """
        Draw sparse grayscale OHLC bar chart.

        Key features:
        - Fixed image sizes: {5: (32,15), 20: (64,60), 60: (96,180)}
        - Bar width: 3px, NO centering — bars fill from x=0
        - Volume at TOP (1/5 of height), price below with 1px gap
        - Grayscale uint8: black bg (0), white bars (255), gray volume (200)
        - Drawing: center pixel = high-low line, left tick = open, right tick = close

        Args:
            open_p, high_p, low_p, close_p: OHLC price arrays
            volume: Volume array
            window_size: Number of trading days (5, 20, or 60)

        Returns:
            (H, W) uint8 ndarray — single-channel grayscale image
        """
        H, W = self.SPARSE_IMAGE_SIZES.get(window_size, (64, window_size * 3))

        img = np.zeros((H, W), dtype=np.uint8)

        num_bars = len(open_p)

        # Volume section: top 1/5 of image
        vol_height = H // 5
        # 1px gap
        price_start_y = vol_height + 1
        price_height = H - price_start_y

        if price_height < 3:
            return img

        # --- Draw volume bars at TOP ---
        if len(volume) > 0:
            max_vol = np.max(volume)
            if max_vol < 1e-8:
                max_vol = 1.0

            for i in range(num_bars):
                x_base = i * 3
                if x_base + 2 >= W:
                    break

                vol_norm = volume[i] / max_vol
                bar_h = max(0, int(vol_norm * (vol_height - 1)))
                if bar_h > 0:
                    y_top = vol_height - bar_h
                    y_bottom = vol_height - 1
                    # 3px wide gray volume bar
                    img[y_top:y_bottom + 1, x_base:x_base + 3] = 200

        # --- Draw price OHLC bars below volume ---
        all_prices = np.concatenate([open_p, high_p, low_p, close_p])
        min_price = np.min(all_prices)
        max_price = np.max(all_prices)
        price_range = max_price - min_price
        if price_range < 1e-8:
            price_range = 1e-8

        # Small padding (1px) at top and bottom of price area
        pad = 1
        effective_height = price_height - 2 * pad

        def price_to_y(price):
            normalized = (price - min_price) / price_range
            # High price at top (low y), low price at bottom (high y)
            y = price_start_y + pad + int((1 - normalized) * effective_height)
            return max(price_start_y + pad, min(H - pad - 1, y))

        for i in range(num_bars):
            x_base = i * 3
            if x_base + 2 >= W:
                break

            y_high = price_to_y(high_p[i])
            y_low = price_to_y(low_p[i])
            y_open = price_to_y(open_p[i])
            y_close = price_to_y(close_p[i])

            # Center pixel (x_base + 1): high-low vertical line
            y_top = min(y_high, y_low)
            y_bot = max(y_high, y_low)
            img[y_top:y_bot + 1, x_base + 1] = 255

            # Left tick (x_base): open price
            img[y_open, x_base] = 255
            img[y_open, x_base + 1] = 255

            # Right tick (x_base + 2): close price
            img[y_close, x_base + 1] = 255
            img[y_close, x_base + 2] = 255

        return img

    def create_gaf_ohlc(self, open_p, high_p, low_p, close_p, volume,
                        size=None, method='gasf', combine_channels=True):
        """
        Create multi-channel GAF image from OHLC data.
        
        Based on Chen & Tsai (2020): "Encoding candlesticks as images for 
        pattern classification using convolutional neural networks"
        
        Args:
            open_p, high_p, low_p, close_p: OHLC arrays
            volume: Volume array
            size: Image size (H, W), default 256x256
            method: 'gasf' or 'gadf'
            combine_channels: If True, stack OHLC+Volume as 5 channels
                            If False, return single averaged GAF
        
        Returns:
            GAF image: either 3-channel RGB or 5-channel (O,H,L,C,V)
        """
        if size is None:
            size = self.img_size
        H, W = size
        
        # Resample all series to target size using the window size
        from scipy.interpolate import interp1d
        
        def resample_to(series, target_len):
            if len(series) == target_len:
                return series
            x_old = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_len)
            f = interp1d(x_old, series, kind='linear', bounds_error=False, fill_value='extrapolate')
            return f(x_new)
        
        target_len = H  # Square image
        
        # Resample each component
        o_resampled = resample_to(open_p, target_len)
        h_resampled = resample_to(high_p, target_len)
        l_resampled = resample_to(low_p, target_len)
        c_resampled = resample_to(close_p, target_len)
        v_resampled = resample_to(volume, target_len) if len(volume) > 0 else np.zeros(target_len)
        
        # Compute GAF for each channel
        def compute_gaf(series):
            # Normalize to [-1, 1]
            min_val = np.min(series)
            max_val = np.max(series)
            if max_val - min_val < 1e-8:
                scaled = np.zeros_like(series)
            else:
                scaled = (2 * series - max_val - min_val) / (max_val - min_val)
            scaled = np.clip(scaled, -1, 1)
            
            # Polar encoding
            phi = np.arccos(scaled)
            
            # GASF: cos(phi_i + phi_j)
            if method == 'gasf':
                gaf = np.cos(phi[:, None] + phi[None, :])
            else:  # GADF
                gaf = np.sin(phi[:, None] - phi[None, :])
            
            # Normalize to [0, 255] for image
            gaf = (gaf + 1) / 2 * 255
            return gaf.astype(np.uint8)
        
        if combine_channels:
            # Create 5-channel representation (can be split or averaged)
            # For CNN input, we'll use the first 3 as pseudo-RGB
            # or stack them properly in the model
            gaf_o = compute_gaf(o_resampled)
            gaf_h = compute_gaf(h_resampled)
            gaf_l = compute_gaf(l_resampled)
            gaf_c = compute_gaf(c_resampled)
            gaf_v = compute_gaf(v_resampled)
            
            # Stack as 5-channel image (H, W, 5)
            # Model will need to handle this appropriately
            gaf_multi = np.stack([gaf_o, gaf_h, gaf_l, gaf_c, gaf_v], axis=-1)
            return gaf_multi
        else:
            # Average all components into single GAF
            avg_series = (o_resampled + h_resampled + l_resampled + c_resampled) / 4
            gaf = compute_gaf(avg_series)
            # Convert to 3-channel RGB
            return np.stack([gaf, gaf, gaf], axis=-1)

    def draw_hybrid(self, open_p, high_p, low_p, close_p, volume,
                    size=None, mode='ohlc_gaf'):
        """
        Hybrid representation: OHLC bars + GAF encoding.
        
        Args:
            mode: 'ohlc' - pure OHLC bars
                  'gaf' - pure GAF encoding
                  'ohlc_gaf' - OHLC in RGB + GAF in alpha/combined
        """
        if mode == 'ohlc':
            return self.draw_ohlc_bars(open_p, high_p, low_p, close_p, volume, size)
        elif mode == 'gaf':
            return self.create_gaf_ohlc(open_p, high_p, low_p, close_p, volume, size)
        elif mode == 'ohlc_gaf':
            # Create side-by-side or overlaid representation
            ohlc = self.draw_ohlc_bars(open_p, high_p, low_p, close_p, volume, size)
            gaf = self.create_gaf_ohlc(open_p, high_p, low_p, close_p, volume, size, combine_channels=False)
            # Blend them (weighted average)
            hybrid = (0.6 * ohlc.astype(float) + 0.4 * gaf.astype(float)).astype(np.uint8)
            return hybrid
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def draw_enhanced(self, open_p, high_p, low_p, close_p, volume,
                      size=None, grayscale=False,
                      use_clahe=True, use_blur=False,
                      output_channels='rgb',
                      chart_type='candle'):
        """
        Draw K-line image with enhanced preprocessing pipeline.

        Args:
            open_p, high_p, low_p, close_p: OHLC arrays
            volume: Volume array
            size: Image size
            grayscale: Output grayscale
            use_clahe: Apply CLAHE enhancement
            use_blur: Apply Gaussian blur
            output_channels: 'rgb', 'rgb+edge', 'rgb+hsv', 'all', 'gaf', 'ohlc'
            chart_type: 'candle', 'ohlc', 'gaf', 'hybrid'

        Returns:
            Processed image with specified channels
        """
        # Route to appropriate drawing method
        if chart_type == 'ohlc':
            img = self.draw_ohlc_bars(open_p, high_p, low_p, close_p, volume, size, grayscale)
        elif chart_type == 'gaf':
            img = self.create_gaf_ohlc(open_p, high_p, low_p, close_p, volume, size, combine_channels=False)
            if grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return img
        elif chart_type == 'hybrid':
            img = self.draw_hybrid(open_p, high_p, low_p, close_p, volume, size, mode='ohlc_gaf')
        else:
            # Default candlestick
            img = self.fast_cv2_draw(open_p, high_p, low_p, close_p, volume, size, grayscale)

        # Apply blur if requested (noise reduction)
        if use_blur:
            img = self.apply_gaussian_blur(img, kernel_size=3)

        # Apply CLAHE if requested (contrast enhancement)
        if use_clahe and not grayscale and len(img.shape) == 3:
            img = self.apply_clahe(img)

        # Add extra channels based on output_channels
        if output_channels == 'rgb' or output_channels == 'ohlc':
            return img
        elif output_channels == 'rgb+edge':
            return self.create_multi_channel(img, include_edge=True, include_hsv_h=False)
        elif output_channels == 'rgb+hsv':
            return self.create_multi_channel(img, include_edge=False, include_hsv_h=True)
        elif output_channels == 'all':
            return self.create_multi_channel(img, include_edge=True, include_hsv_h=True)
        else:
            return img

