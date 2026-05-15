#!/usr/bin/env bash
#SBATCH --job-name=sono-ana
#SBATCH --account=mai-ws-ai-infra
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/vast/exp/brucexu/EchoJEPA/logs/ana-%x-%j.out
#SBATCH --error=/mnt/vast/exp/brucexu/EchoJEPA/logs/ana-%x-%j.err
#
# Run ONE analysis script on ONE checkpoint.
# Inputs (env):
#   SCRIPT     : analysis module name (e.g. forecast_curves)
#   CHECKPOINT : path to the trained .pt
#   OUT        : output directory
#   EXTRA      : extra CLI args (already-substituted, single string)
#   REPO       : repo root (default: $SLURM_SUBMIT_DIR)
set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
SCRIPT=${SCRIPT:?SCRIPT env var required}
CHECKPOINT=${CHECKPOINT:?CHECKPOINT env var required}
OUT=${OUT:?OUT env var required}
EXTRA=${EXTRA:-}
VENV=${VENV:-/mnt/vast/exp/brucexu/EchoJEPA/.venv}

source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO"
export HOME=${HOME:-/tmp}
mkdir -p "$OUT"

echo "[$(date)] node=$(hostname) jobid=$SLURM_JOB_ID script=$SCRIPT ckpt=$CHECKPOINT out=$OUT"
nvidia-smi -L

srun python3 -m experiments.analysis.${SCRIPT} \
        --checkpoint "$CHECKPOINT" \
        --out "$OUT" \
        $EXTRA
