"""
Benchmark script for image rendering performance.
Tests the vectorized drawing functions for correctness and speed.

Usage:
    python scripts/benchmark_rendering.py
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.image_generator import ImageGenerator


def generate_sample_data(num_candles=20, seed=42):
    """Generate synthetic OHLCV data for benchmarking."""
    np.random.seed(seed)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(num_candles) * 0.5)
    
    opens = prices + np.random.randn(num_candles) * 0.3
    closes = prices + np.random.randn(num_candles) * 0.3
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(num_candles) * 0.5)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(num_candles) * 0.5)
    volumes = np.abs(np.random.randn(num_candles) * 1000000 + 5000000)
    
    return opens, highs, lows, closes, volumes


def benchmark_function(func, args, kwargs, iterations=500, name=""):
    """Benchmark a function over N iterations."""
    # Warmup
    for _ in range(5):
        result = func(*args, **kwargs)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / iterations) * 1000
    print(f"  {name}: {avg_ms:.3f} ms/call ({iterations} iterations, {elapsed:.2f}s total)")
    return result, avg_ms


def test_draw_ohlc_bars(img_gen, ohlcv):
    """Test draw_ohlc_bars for correctness and speed."""
    opens, highs, lows, closes, volumes = ohlcv
    
    print("\n=== draw_ohlc_bars ===")
    
    # Test grayscale
    img_gray, ms_gray = benchmark_function(
        img_gen.draw_ohlc_bars, 
        (opens, highs, lows, closes, volumes),
        {"size": (64, 60), "grayscale": True, "include_volume": True},
        name="grayscale 64x60"
    )
    
    # Test RGB
    img_rgb, ms_rgb = benchmark_function(
        img_gen.draw_ohlc_bars,
        (opens, highs, lows, closes, volumes),
        {"size": (256, 256), "grayscale": False, "include_volume": True},
        name="RGB 256x256"
    )
    
    # Correctness checks
    assert img_gray.shape == (64, 60), f"Expected (64, 60), got {img_gray.shape}"
    assert img_gray.dtype == np.uint8
    assert img_gray.max() > 0, "Image is all black (no drawing occurred)"
    
    assert img_rgb.shape == (256, 256, 3), f"Expected (256, 256, 3), got {img_rgb.shape}"
    assert img_rgb.dtype == np.uint8
    assert img_rgb.max() > 0, "RGB image is all black"
    
    print(f"  ✓ Correctness checks passed")
    print(f"  Gray pixels lit: {np.count_nonzero(img_gray)} / {img_gray.size}")
    print(f"  RGB pixels lit: {np.count_nonzero(img_rgb.sum(axis=2))} / {img_rgb.shape[0]*img_rgb.shape[1]}")
    

def test_fast_cv2_draw(img_gen, ohlcv):
    """Test fast_cv2_draw for correctness and speed."""
    opens, highs, lows, closes, volumes = ohlcv
    
    print("\n=== fast_cv2_draw ===")
    
    img_rgb, ms = benchmark_function(
        img_gen.fast_cv2_draw,
        (opens, highs, lows, closes, volumes),
        {"size": (128, 128), "grayscale": False},
        name="RGB 128x128"
    )
    
    img_gray, ms_gray = benchmark_function(
        img_gen.fast_cv2_draw,
        (opens, highs, lows, closes, volumes),
        {"size": (64, 64), "grayscale": True},
        name="grayscale 64x64"
    )
    
    assert img_rgb.shape == (128, 128, 3), f"Expected (128, 128, 3), got {img_rgb.shape}"
    assert img_rgb.max() > 0, "Image is all black"
    assert img_gray.shape == (64, 64, 1), f"Expected (64, 64, 1), got {img_gray.shape}"
    assert img_gray.max() > 0, "Gray image is all black"
    
    print(f"  ✓ Correctness checks passed")


def test_draw_sparse_ohlc(img_gen, ohlcv):
    """Test draw_sparse_ohlc for correctness and speed."""
    opens, highs, lows, closes, volumes = ohlcv
    
    print("\n=== draw_sparse_ohlc ===")
    
    for win_size in [5, 20, 60]:
        n = min(win_size, len(opens))
        o, h, l, c, v = opens[:n], highs[:n], lows[:n], closes[:n], volumes[:n]
        
        expected_size = ImageGenerator.SPARSE_IMAGE_SIZES.get(win_size, (64, win_size * 3))
        
        img, ms = benchmark_function(
            img_gen.draw_sparse_ohlc,
            (o, h, l, c, v),
            {"window_size": win_size},
            name=f"window={win_size} ({expected_size})"
        )
        
        assert img.shape == expected_size, f"Expected {expected_size}, got {img.shape}"
        assert img.dtype == np.uint8
        if n > 0:
            assert img.max() > 0, f"Image is all black for window_size={win_size}"
    
    print(f"  ✓ Correctness checks passed for all window sizes")


def test_edge_cases(img_gen):
    """Test edge cases."""
    print("\n=== Edge Cases ===")
    
    # Flat prices (price_range == 0)
    flat = np.ones(20) * 100.0
    vol = np.ones(20) * 1000000.0
    img = img_gen.draw_ohlc_bars(flat, flat, flat, flat, vol, size=(64, 60), grayscale=True)
    assert img.shape == (64, 60), "Flat prices broke draw_ohlc_bars"
    print("  ✓ Flat prices handled correctly")
    
    # Single bar
    one = np.array([100.0])
    vol1 = np.array([1000000.0])
    img = img_gen.draw_sparse_ohlc(one, one + 1, one - 1, one + 0.5, vol1, window_size=5)
    assert img.shape == (32, 15), "Single bar broke draw_sparse_ohlc"
    print("  ✓ Single bar handled correctly")
    
    # Empty volume
    opens, highs, lows, closes, _ = generate_sample_data(20)
    img = img_gen.draw_ohlc_bars(opens, highs, lows, closes, np.array([]),
                                  size=(64, 60), grayscale=True, include_volume=False)
    assert img.shape == (64, 60), "Empty volume broke draw_ohlc_bars"
    print("  ✓ Empty volume handled correctly")
    
    print("  ✓ All edge cases passed")


def main():
    print("=" * 60)
    print("Image Rendering Benchmark (Vectorized)")
    print("=" * 60)
    
    img_gen = ImageGenerator(img_size=(128, 128), norm_method='robust', include_volume=True)
    
    # Generate data for different window sizes
    ohlcv_20 = generate_sample_data(20)
    ohlcv_60 = generate_sample_data(60)
    
    # Run benchmarks
    test_draw_ohlc_bars(img_gen, ohlcv_20)
    test_fast_cv2_draw(img_gen, ohlcv_20)
    test_draw_sparse_ohlc(img_gen, ohlcv_60)
    test_edge_cases(img_gen)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
