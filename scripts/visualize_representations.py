"""
Visualize different K-line image representations.
Compare: Candlestick, OHLC bars, GAF encoding

Usage:
    python scripts/visualize_representations.py --symbol AAPL --market us
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import argparse
from src.data.image_generator import ImageGenerator


def load_stock_data(symbol, market='us'):
    """Load stock data from CSV."""
    data_dir = PROJECT_ROOT / 'data' / 'raw' / market
    csv_file = data_dir / f"{symbol}.csv"
    
    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        return None
    
    df = pd.read_csv(csv_file)
    df.columns = [col.title() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def create_comparison_figure(df, window_size=10, start_idx=-1):
    """Create comparison figure of different representations."""
    
    if start_idx < 0:
        # Use last complete window
        start_idx = len(df) - window_size
    
    end_idx = start_idx + window_size
    window_df = df.iloc[start_idx:end_idx]
    
    opens = window_df['Open'].values.astype(np.float64)
    highs = window_df['High'].values.astype(np.float64)
    lows = window_df['Low'].values.astype(np.float64)
    closes = window_df['Close'].values.astype(np.float64)
    vols = window_df['Volume'].values.astype(np.float64)
    
    # Create image generators
    img_gen_256 = ImageGenerator(img_size=(256, 256), norm_method='robust')
    img_gen_128 = ImageGenerator(img_size=(128, 128), norm_method='robust')
    
    # Generate different representations
    print("Generating representations...")
    
    # 1. Original candlestick (128x128, 20 days - baseline)
    candle_20d = img_gen_128.fast_cv2_draw(
        df['Open'].iloc[start_idx-10:start_idx+10].values,
        df['High'].iloc[start_idx-10:start_idx+10].values,
        df['Low'].iloc[start_idx-10:start_idx+10].values,
        df['Close'].iloc[start_idx-10:start_idx+10].values,
        df['Volume'].iloc[start_idx-10:start_idx+10].values,
        size=(128, 128),
        grayscale=False
    )
    
    # 2. Candlestick (256x256, 10 days)
    candle_10d = img_gen_256.fast_cv2_draw(
        opens, highs, lows, closes, vols,
        size=(256, 256),
        grayscale=False
    )
    
    # 3. OHLC bars (256x256, 10 days) - Xiu et al. style
    ohlc_10d = img_gen_256.draw_ohlc_bars(
        opens, highs, lows, closes, vols,
        size=(256, 256),
        grayscale=False,
        include_volume=True
    )
    
    # 4. GAF encoding (256x256, 10 days) - Chen & Tsai style
    gaf_10d = img_gen_256.create_gaf_ohlc(
        opens, highs, lows, closes, vols,
        size=(256, 256),
        method='gasf',
        combine_channels=False
    )
    
    # 5. Hybrid
    hybrid_10d = img_gen_256.draw_hybrid(
        opens, highs, lows, closes, vols,
        size=(256, 256),
        mode='ohlc_gaf'
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'K-Line Representations Comparison\n{window_size}-Day Window', fontsize=14, fontweight='bold')
    
    # Plot 1: Candlestick 20d/128px (baseline)
    axes[0, 0].imshow(cv2.cvtColor(candle_20d, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Baseline: Candlestick\n20 days, 128×128', fontsize=10)
    axes[0, 0].axis('off')
    
    # Plot 2: Candlestick 10d/256px
    axes[0, 1].imshow(cv2.cvtColor(candle_10d, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Candlestick\n10 days, 256×256', fontsize=10)
    axes[0, 1].axis('off')
    
    # Plot 3: OHLC bars 10d/256px
    axes[0, 2].imshow(cv2.cvtColor(ohlc_10d, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('OHLC Bars (Xiu et al. 2021)\n10 days, 256×256', fontsize=10)
    axes[0, 2].axis('off')
    
    # Plot 4: GAF 10d/256px
    axes[1, 0].imshow(gaf_10d)
    axes[1, 0].set_title('GAF Encoding (Chen & Tsai 2020)\n10 days, 256×256', fontsize=10)
    axes[1, 0].axis('off')
    
    # Plot 5: Hybrid
    axes[1, 1].imshow(cv2.cvtColor(hybrid_10d, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Hybrid (OHLC + GAF)\n10 days, 256×256', fontsize=10)
    axes[1, 1].axis('off')
    
    # Plot 6: Information comparison
    axes[1, 2].axis('off')
    info_text = """
    Key Differences:
    
    Candlestick (Baseline):
    - 20 days @ 128×128
    - ~6px per candle
    - Standard representation
    
    OHLC Bars (Xiu et al.):
    - 10 days @ 256×256
    - 3px per bar (open, high-low, close)
    - Sparse representation
    - Normalized scale across stocks
    
    GAF (Chen & Tsai):
    - Polar coordinate encoding
    - Preserves temporal correlations
    - Gramian Angular Summation Field
    - 90.7% accuracy in pattern recognition
    
    Expected Improvements:
    - Higher resolution per candle
    - Better cross-stock normalization
    - Captured temporal patterns
    """
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=9, verticalalignment='center',
                   family='monospace', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = PROJECT_ROOT / 'outputs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'representation_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Also save individual images for closer inspection
    cv2.imwrite(str(output_dir / '01_candle_20d_128.png'), candle_20d)
    cv2.imwrite(str(output_dir / '02_candle_10d_256.png'), candle_10d)
    cv2.imwrite(str(output_dir / '03_ohlc_10d_256.png'), ohlc_10d)
    cv2.imwrite(str(output_dir / '04_gaf_10d_256.png'), gaf_10d)
    cv2.imwrite(str(output_dir / '05_hybrid_10d_256.png'), hybrid_10d)
    print(f"Individual images saved to: {output_dir}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize K-line representations')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--market', type=str, default='us', help='Market (us, cn, etc.)')
    parser.add_argument('--window-size', type=int, default=10, help='Window size (default: 10)')
    parser.add_argument('--start-idx', type=int, default=-1, help='Start index (-1 for last window)')
    args = parser.parse_args()
    
    print(f"Loading data for {args.symbol}...")
    df = load_stock_data(args.symbol, args.market)
    
    if df is None:
        return
    
    print(f"Data loaded: {len(df)} days")
    print(f"Creating comparison visualization...")
    
    create_comparison_figure(df, window_size=args.window_size, start_idx=args.start_idx)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
