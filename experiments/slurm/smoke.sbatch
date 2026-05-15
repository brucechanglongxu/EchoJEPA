#!/usr/bin/env bash
#SBATCH --job-name=sono-smoke
#SBATCH --account=mai-ws-ai-infra
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/vast/exp/brucexu/EchoJEPA/logs/smoke-%j.out
#SBATCH --error=/mnt/vast/exp/brucexu/EchoJEPA/logs/smoke-%j.err
#
# Smoke test: verify Slurm wiring, GPU visibility, repo imports, and
# that the SonoState modules + V-JEPA-2 checkpoint can be loaded.
# Does NOT need EchoNet data.
set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
CKPT=${CKPT:-/mnt/vast/checkpoints/brucexu/vjepa2/vitl.pt}

cd "$REPO"
export PYTHONPATH="$REPO"
export HOME=${HOME:-/tmp}

echo "=== env ==="
echo "host=$(hostname)  jobid=$SLURM_JOB_ID  cwd=$PWD"
echo "python=$(which python3)  $(python3 --version)"
nvidia-smi -L
echo

echo "=== imports ==="
srun python3 - <<PY
import os, sys, time, torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "ngpu", torch.cuda.device_count(),
      "dev0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# Import repo modules
from src.models.sonostate import StateHead, Transition
from src.masks.sonostate_collator import SonoStateCollator
print("imported StateHead, Transition, SonoStateCollator OK")

# Build modules + a small forward pass
embed_dim, state_dim = 1024, 256
sh = StateHead(embed_dim=embed_dim, state_dim=state_dim).cuda()
tr = Transition(state_dim=state_dim, hidden_dim=512,
                zero_init=True, residual=True, init_scale=-4.6).cuda()

B, T, D = 2, 4, embed_dim
toks = torch.randn(B, T, D, device="cuda")
z = sh(toks); z2 = tr(z)
print("StateHead out:", tuple(z.shape), "norm=", z.norm(dim=-1).mean().item())
print("Transition step delta:", (z2 - z).abs().mean().item(),
      "(should be ~0 with identity init)")

# Try loading the V-JEPA-2 checkpoint header
ck = "${CKPT}"
print("checkpoint:", ck, "size_MB=", os.path.getsize(ck)/2**20)
sd = torch.load(ck, map_location="cpu", weights_only=False)
keys = list(sd.keys()) if isinstance(sd, dict) else ["<not a dict>"]
print("ckpt top-level keys:", keys[:8], "..." if len(keys) > 8 else "")
PY

echo
echo "=== done ==="
