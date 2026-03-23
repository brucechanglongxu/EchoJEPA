#!/bin/bash
# SonoState training with FROZEN pretrained encoder
# Only trains: predictor (22M), state_head (264K), transition (263K)
# No inf gradient issues since encoder backward is skipped

cd /scratch/bxu/project/EchoJEPA

export PYTHONPATH=/scratch/bxu/project/EchoJEPA
export HOME=/tmp
export MIOPEN_USER_DB_PATH=/tmp/miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache
mkdir -p /tmp/miopen-cache

pip install eva-decord --quiet --break-system-packages 2>/dev/null || pip install eva-decord --quiet 2>/dev/null

OUTDIR=/scratch/bxu/project/EchoJEPA/checkpoints/sonostate_frozen
mkdir -p "$OUTDIR"

pkill -9 -f "app.main.*sonostate" 2>/dev/null
sleep 2

nohup python3 -m app.main \
    --fname configs/train/vitl16/sonostate-frozen-encoder.yaml \
    --devices cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    > "$OUTDIR/train_stdout.log" 2>&1 &

echo "SonoState FROZEN-ENCODER training launched with PID: $!"
echo "Output dir: $OUTDIR"
echo "Monitor with: tail -f $OUTDIR/train_stdout.log"
