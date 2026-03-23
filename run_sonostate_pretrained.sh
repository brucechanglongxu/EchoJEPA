#!/bin/bash
# SonoState overnight training: pretrained V-JEPA ViT-L encoder on EchoNet-Dynamic
# 7 GPUs (skipping GPU 0), bfloat16, gradient sanitization

cd /scratch/bxu/project/EchoJEPA

export PYTHONPATH=/scratch/bxu/project/EchoJEPA
export HOME=/tmp
export MIOPEN_USER_DB_PATH=/tmp/miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache
mkdir -p /tmp/miopen-cache

# Auto-install dependencies that may be missing in fresh containers
pip install eva-decord --quiet --break-system-packages 2>/dev/null || pip install eva-decord --quiet 2>/dev/null

OUTDIR=/scratch/bxu/project/EchoJEPA/checkpoints/sonostate_pretrained_fp32
mkdir -p "$OUTDIR"

# Kill any lingering training processes
pkill -9 -f "app.main.*sonostate" 2>/dev/null
sleep 2

nohup python3 -m app.main \
    --fname configs/train/vitl16/sonostate-pretrained-overnight.yaml \
    --devices cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    > "$OUTDIR/train_stdout.log" 2>&1 &

echo "SonoState pretrained training launched with PID: $!"
echo "Output dir: $OUTDIR"
echo "Monitor with: tail -f $OUTDIR/train_stdout.log"
