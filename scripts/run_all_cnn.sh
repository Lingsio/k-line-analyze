#!/bin/bash
# Run all 4 CNN experiments sequentially
# Usage: bash scripts/run_all_cnn.sh

set -e
cd /workspace

echo "========================================"
echo "Running all CNN experiments sequentially"
echo "========================================"
echo "Start time: $(date)"
echo ""

echo "[1/4] Running CNN-Raw..."
python scripts/run_cnn_raw.py 2>&1 | tee outputs/eccv_results/cnn_raw.log
echo ""

echo "[2/4] Running CNN-Basic..."
python scripts/run_cnn_basic.py 2>&1 | tee outputs/eccv_results/cnn_basic.log
echo ""

echo "[3/4] Running KLineNet..."
python scripts/run_klinenet.py 2>&1 | tee outputs/eccv_results/klinenet.log
echo ""

echo "[4/4] Running KLineNet-MC..."
python scripts/run_klinenet_mc.py 2>&1 | tee outputs/eccv_results/klinenet_mc.log
echo ""

echo "========================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "========================================"
