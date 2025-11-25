#!/bin/bash
# Sequential training: FULL then SIMPLE for Band-Split VAE

echo "========================================"
echo "Band-Split VAE Sequential Training"
echo "========================================"

# Create directories
mkdir -p runs/bandvae_full
mkdir -p runs/bandvae_simple

# Train FULL version
echo ""
echo "[1/2] Training FULL version (320 channels per band, 3 bands)..."
echo "Log: runs/bandvae_full/train.log"
python3 -u train_bandvae.py --mode full --epochs 20 --batch_size 64 --device cuda > runs/bandvae_full/train.log 2>&1

FULL_EXIT_CODE=$?
echo ""
echo "FULL training completed with exit code: $FULL_EXIT_CODE"

if [ $FULL_EXIT_CODE -eq 0 ]; then
    echo "✓ FULL training succeeded"
else
    echo "✗ FULL training failed"
fi

# Train SIMPLE version
echo ""
echo "[2/2] Training SIMPLE version (160 channels per band, 3 bands)..."
echo "Log: runs/bandvae_simple/train.log"
python3 -u train_bandvae.py --mode simple --epochs 20 --batch_size 64 --device cuda > runs/bandvae_simple/train.log 2>&1

SIMPLE_EXIT_CODE=$?
echo ""
echo "SIMPLE training completed with exit code: $SIMPLE_EXIT_CODE"

if [ $SIMPLE_EXIT_CODE -eq 0 ]; then
    echo "✓ SIMPLE training succeeded"
else
    echo "✗ SIMPLE training failed"
fi

# Summary
echo ""
echo "========================================"
echo "Training Summary"
echo "========================================"
echo "FULL version:   $([ $FULL_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "SIMPLE version: $([ $SIMPLE_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo ""
echo "Logs:"
echo "  FULL:   runs/bandvae_full/train.log"
echo "  SIMPLE: runs/bandvae_simple/train.log"
echo ""
echo "Models:"
echo "  FULL:   runs/bandvae_full/best.pt"
echo "  SIMPLE: runs/bandvae_simple/best.pt"
echo "========================================"
