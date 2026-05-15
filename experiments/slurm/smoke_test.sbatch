#!/usr/bin/env bash
#SBATCH --job-name=sono-smoke
#SBATCH --account=mai-ws-ai-infra
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/vast/exp/brucexu/EchoJEPA/logs/smoke-%j.out
#SBATCH --error=/mnt/vast/exp/brucexu/EchoJEPA/logs/smoke-%j.err
#
# Smoke test: verify Slurm wiring + venv + CUDA + SonoState imports.
# Does NOT touch EchoNet or train anything.
set -euo pipefail

REPO=${REPO:-/mnt/home/brucexu/nips/EchoJEPA}
VENV=/mnt/vast/exp/brucexu/EchoJEPA/.venv

echo "[$(date)] node=$(hostname) jobid=$SLURM_JOB_ID"
echo "SLURM allocation:"
echo "  CPUs:    $SLURM_CPUS_ON_NODE"
echo "  GPUs:    ${SLURM_GPUS_ON_NODE:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Memory:  $SLURM_MEM_PER_NODE MB"
nvidia-smi -L

source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO"

python - <<'PY'
import torch, decord, yaml, einops, timm
print("torch", torch.__version__, "cuda built:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available(),
      "device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))

from src.models.sonostate import StateHead, Transition
from src.masks.sonostate_collator import SonoStateCollator  # noqa: F401
import app.sonostate.train  # noqa: F401
import app.sonostate.utils  # noqa: F401

sh = StateHead(1024, 256).cuda()
tr = Transition(256, 512, zero_init=True, residual=True, init_scale=-4.6).cuda()
z  = torch.randn(2, 16, 1024, device="cuda")
s  = sh(z)                       # (B, d=256)
n  = tr(s)                       # (B, d=256)
print("StateHead out:", s.shape, "Transition out:", n.shape)
print("Smoke test OK.")
PY

echo "[$(date)] smoke test completed."
