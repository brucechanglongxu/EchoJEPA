#!/usr/bin/env bash
#SBATCH --job-name=sono-train
#SBATCH --account=mai-ws-ai-infra
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast/exp/brucexu/EchoJEPA/logs/train-%x-%A_%a.out
#SBATCH --error=/mnt/vast/exp/brucexu/EchoJEPA/logs/train-%x-%A_%a.err
#
# Train ONE SonoState cell.
# Inputs (env, set by submit script or via --export):
#   CONFIG : absolute path to a generated YAML
#   REPO   : path to EchoJEPA repo (default: $SLURM_SUBMIT_DIR)
set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
CONFIG=${CONFIG:?CONFIG env var required}
VENV=${VENV:-/mnt/vast/exp/brucexu/EchoJEPA/.venv}

source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO"
export HOME=${HOME:-/tmp}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-/tmp/miopen-cache-$USER-$SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Read folder from YAML so we know where to drop the stdout log.
FOLDER=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['folder'])")
mkdir -p "$FOLDER"
LOG="$FOLDER/train_stdout.log"

GPU_LIST="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
NGPU=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}
GPU_LIST=$(python3 -c "print(' '.join(f'cuda:{i}' for i in range($NGPU)))")
echo "[$(date)] node=$(hostname) jobid=$SLURM_JOB_ID ngpu=$NGPU config=$CONFIG -> $FOLDER" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

srun --output="$LOG" --error="$LOG" --open-mode=append \
    python3 -m app.main \
        --fname "$CONFIG" \
        --devices $GPU_LIST

# Guard: app.main swallows exceptions and exits 0. Refuse success unless
# at least one real checkpoint landed, so dependent analyses don't fire on
# a half-broken cell.
shopt -s nullglob
ckpts=( "$FOLDER"/e*.pt "$FOLDER"/latest.pt )
if (( ${#ckpts[@]} == 0 )); then
    echo "[$(date)] FATAL: no checkpoint produced in $FOLDER" | tee -a "$LOG"
    exit 42
fi
echo "[$(date)] OK: produced ${#ckpts[@]} checkpoint(s) in $FOLDER" | tee -a "$LOG"
