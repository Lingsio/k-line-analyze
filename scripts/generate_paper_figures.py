"""
Generate figures for ECCV 2026 paper: SAK-Net

Usage:
    python scripts/generate_paper_figures.py --output paper/figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import argparse

# Set matplotlib style for ECCV paper
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8


def create_placeholder_figure():
    """Create a placeholder figure with instructions"""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, 'Placeholder\nReplace with actual figure', 
            ha='center', va='center', fontsize=14, color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig


def generate_figure1_encoding_comparison(output_dir):
    """
    Figure 1: Encoding methods comparison
    Shows Candlestick, OHLC Bars, GAF, and Hybrid encoding
    """
    print("Generating Figure 1: Encoding comparison...")
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    # Placeholder for each encoding type
    for idx, (ax, title) in enumerate(zip(axes, ['Candlestick RGB', 'OHLC Bars', 'GAF', 'Hybrid'])):
        ax.set_title(f"({chr(97+idx)}) {title}", fontweight='bold')
        ax.text(0.5, 0.5, f'{title}\n(20-day window)', 
                ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'encoding_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'encoding_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/encoding_comparison.pdf")


def generate_figure2_sector_comparison(output_dir):
    """
    Figure 2: Sector-adaptive results comparison
    Bar chart showing accuracy for each sector
    """
    print("Generating Figure 2: Sector comparison...")
    
    sectors = ['Consumer', 'Industrials-\nEnergy', 'Tech-\nSemiconductors', 
               'Tech-\nSoftware', 'Financials', 'Healthcare']
    accuracies = [60.37, 59.94, 59.88, 57.22, 55.90, 53.54]
    colors = ['#0066CC' if acc > 57 else '#6699CC' for acc in accuracies]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(range(len(sectors)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add baseline lines
    ax.axhline(y=51.26, color='gray', linestyle='--', linewidth=1.5, label='Universal CNN (51.26%)')
    ax.axhline(y=50.00, color='red', linestyle=':', linewidth=1.5, label='Random (50.00%)')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Industry Sector', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Sector-Adaptive Training Results', fontweight='bold', fontsize=11)
    ax.set_xticks(range(len(sectors)))
    ax.set_xticklabels(sectors, rotation=0, ha='center')
    ax.set_ylim(48, 62)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sector_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sector_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/sector_comparison.pdf")


def generate_figure3_gradcam(output_dir):
    """
    Figure 3: Grad-CAM visualization
    Shows attention maps for successful and failed predictions
    """
    print("Generating Figure 3: Grad-CAM visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    for idx, (ax, title) in enumerate(zip(axes, ['Successful Prediction', 'Failed Prediction'])):
        ax.set_title(f"({chr(97+idx)}) {title}", fontweight='bold')
        ax.text(0.5, 0.5, 'Grad-CAM Heatmap\n(Placeholder)', 
                ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'gradcam.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/gradcam.pdf")


def generate_figure4_examples(output_dir):
    """
    Figure 4: Prediction examples
    Shows correctly and incorrectly predicted cases
    """
    print("Generating Figure 4: Prediction examples...")
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    for idx, (ax, title) in enumerate(zip(axes, ['Correctly Predicted Uptrend', 
                                                  'Incorrectly Predicted Downtrend'])):
        ax.set_title(f"({chr(97+idx)}) {title}", fontweight='bold')
        ax.text(0.5, 0.5, 'Price Chart with\nPrediction Point', 
                ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_examples.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'prediction_examples.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/prediction_examples.pdf")


def main():
    parser = argparse.ArgumentParser(description='Generate figures for SAK-Net paper')
    parser.add_argument('--output', type=str, default='paper/figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}\n")
    
    # Generate all figures
    generate_figure1_encoding_comparison(args.output)
    generate_figure2_sector_comparison(args.output)
    generate_figure3_gradcam(args.output)
    generate_figure4_examples(args.output)
    
    print("\n✅ All figures generated successfully!")
    print("\nNote: These are placeholder figures. Please replace them with:")
    print("  - Actual K-line chart images for Figure 1")
    print("  - Real Grad-CAM outputs for Figure 3")
    print("  - Actual prediction examples for Figure 4")


if __name__ == '__main__':
    main()
