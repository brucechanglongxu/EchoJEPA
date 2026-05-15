#!/usr/bin/env bash
# End-to-end submit script (Microsoft AI HPC Slurm via SUNK).
#   1. Generate per-cell training YAMLs from sweep matrix.
#   2. Build a bash submitter that chains train -> analyses with --dependency=afterok.
#   3. Run the submitter (which calls sbatch).
# Idempotent: re-running rebuilds the YAMLs + submitter and resubmits.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
EXP_ROOT=${EXP_ROOT:-/mnt/vast/exp/brucexu/EchoJEPA/experiments_2026}
DATA_ROOT=${DATA_ROOT:-/mnt/vast/data/brucexu/echonet}
DATA_CSV=${DATA_CSV:-$DATA_ROOT/pretrain_annotations.csv}
ANNEAL_CKPT=${ANNEAL_CKPT:-/mnt/vast/checkpoints/brucexu/vjepa2/vitl.pt}
BASE_CFG=${BASE_CFG:-configs/train/vitl16/sonostate-frozen-v3.yaml}
ACCOUNT=${ACCOUNT:-mai-ws-ai-infra}
PARTITION=${PARTITION:-hpc-mid}
ANALYSIS_PARTITION=${ANALYSIS_PARTITION:-$PARTITION}
GEN_DIR="$REPO/experiments/configs/_generated"
SUBMIT_FILE="$REPO/experiments/slurm/_submit.sh"

cd "$REPO"
mkdir -p /mnt/vast/exp/brucexu/EchoJEPA/logs

echo "[1/3] generating sweep configs..."
PYTHONPATH="$REPO" python3 experiments/configs/generate_sweeps.py \
    --base "$BASE_CFG" \
    --out  "$GEN_DIR" \
    --exp-root "$EXP_ROOT" \
    --data-csv "$DATA_CSV" \
    --anneal-ckpt "$ANNEAL_CKPT"

echo "[2/3] building Slurm submitter..."
PYTHONPATH="$REPO" python3 experiments/slurm/build_submit.py \
    --manifest "$GEN_DIR/manifest.json" \
    --out      "$SUBMIT_FILE" \
    --data-root "$DATA_ROOT" \
    --repo "$REPO" \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --analysis-partition "$ANALYSIS_PARTITION"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[3/3] DRY_RUN=1 -- skipping submission. Inspect $SUBMIT_FILE then run it manually."
    exit 0
fi

echo "[3/3] submitting jobs to Slurm..."
bash "$SUBMIT_FILE"
echo
echo "Watch:  squeue -u \$USER"
echo "Then:   python experiments/figures/make_paper_figures.py --root $EXP_ROOT --out paper/_figs"
