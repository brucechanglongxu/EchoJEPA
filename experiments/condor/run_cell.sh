#!/usr/bin/env bash
# Wrapper that runs a SonoState training cell on Condor / SLURM hybrid.
# Inputs (env vars):
#   CONFIG       : path to a generated YAML
#   GPU_LIST     : space-separated CUDA device ids (default: cuda:0..cuda:7)
#   REPO         : path to EchoJEPA repo (default: /scratch/bxu/project/EchoJEPA)
#
# Output:
#   ${folder}/train_stdout.log  (folder is read from the YAML)
set -euo pipefail

REPO=${REPO:-/scratch/bxu/project/EchoJEPA}
CONFIG=${CONFIG:?CONFIG env var required}
GPU_LIST=${GPU_LIST:-"cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"}

cd "$REPO"
export PYTHONPATH="$REPO"
export HOME=/tmp
export MIOPEN_USER_DB_PATH=/tmp/miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache
mkdir -p /tmp/miopen-cache

pip install eva-decord --quiet --break-system-packages 2>/dev/null \
  || pip install eva-decord --quiet 2>/dev/null || true

# Read folder from YAML (yq if available, fall back to python)
FOLDER=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['folder'])")
mkdir -p "$FOLDER"
LOG="$FOLDER/train_stdout.log"

echo "[$(date)] launching $CONFIG -> $FOLDER" | tee -a "$LOG"
exec python3 -m app.main \
    --fname "$CONFIG" \
    --devices $GPU_LIST \
    >> "$LOG" 2>&1
