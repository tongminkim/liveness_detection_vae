#!/bin/bash
# Wait for FULL training to complete, then start SIMPLE training

FULL_PID=284759

echo "========================================"
echo "Auto-Training Scheduler"
echo "========================================"
echo "Waiting for FULL training (PID: $FULL_PID) to complete..."
echo ""

# Wait for FULL training process to finish
while ps -p $FULL_PID > /dev/null 2>&1; do
    sleep 60  # Check every minute
done

echo "FULL training completed!"
echo ""
echo "Starting SIMPLE training..."
echo "========================================"
echo ""

# Create directory for SIMPLE logs
mkdir -p runs/vae_simple

# Start SIMPLE training
python3 -u train.py --mode simple --epochs 20 --batch_size 64 --device cuda > runs/vae_simple/train.log 2>&1

SIMPLE_EXIT_CODE=$?

echo ""
echo "========================================"
echo "SIMPLE training completed!"
echo "Exit code: $SIMPLE_EXIT_CODE"
echo "Log: runs/vae_simple/train.log"
echo "Model: runs/vae_simple/best.pt"
echo "========================================"
