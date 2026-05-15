"""Generate sweep configurations from a base SonoState YAML.

Usage:
    python experiments/configs/generate_sweeps.py \
        --base configs/train/vitl16/sonostate-frozen-v3.yaml \
        --out  experiments/configs/_generated

This produces one YAML per (sweep, cell) and a manifest JSON that the
Condor submit files consume.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml


# -----------------------------------------------------------------------------
# Sweep matrix
# -----------------------------------------------------------------------------

SWEEPS: dict[str, list[dict]] = {
    # S1. Latent dimension --- intrinsic-dim probe
    "S1_dim": [
        {"id": f"d{d}", "patch": {"sonostate": {"state_dim": d,
                                                "transition_hidden_dim": max(2 * d, 128)}}}
        for d in (32, 64, 128, 256, 512)
    ],
    # S2. Uniformity weight --- collapse vs spread on the sphere
    "S2_lambda_u": [
        {"id": f"u{int(l*100):04d}", "patch": {"sonostate": {"lambda_uniform": l}}}
        for l in (0.0, 0.1, 1.0, 5.0)
    ],
    # S3. Forecast loss components
    "S3_forecast_loss": [
        {"id": "l1",     "patch": {"sonostate": {"forecast_l1": True,  "forecast_cos": False}}},
        {"id": "cos",    "patch": {"sonostate": {"forecast_l1": False, "forecast_cos": True}}},
        {"id": "l1cos",  "patch": {"sonostate": {"forecast_l1": True,  "forecast_cos": True}}},
    ],
    # S4. Encoder freezing
    "S4_encoder": [
        {"id": "frozen",  "patch": {"sonostate": {"freeze_encoder": True}}},
        {"id": "ftuned",  "patch": {"sonostate": {"freeze_encoder": False}}},
    ],
    # S5. Transition init
    "S5_init": [
        {"id": "identity",
         "patch": {"sonostate": {"transition_zero_init": True,
                                 "transition_residual": True,
                                 "transition_init_scale": -4.6}}},
        {"id": "random",
         "patch": {"sonostate": {"transition_zero_init": False,
                                 "transition_residual": True,
                                 "transition_init_scale": 0.0}}},
        {"id": "noresidual",
         "patch": {"sonostate": {"transition_zero_init": False,
                                 "transition_residual": False,
                                 "transition_init_scale": 0.0}}},
    ],
    # S6. Schedule
    "S6_schedule": [
        {"id": "ep150",
         "patch": {"optimization": {"epochs": 150, "warmup": 10,
                                    "lr": 3.0e-5, "final_lr": 1.0e-6}}},
        {"id": "ep300_cool",
         "patch": {"optimization": {"epochs": 300, "warmup": 20,
                                    "lr": 3.0e-5, "final_lr": 1.0e-7,
                                    "is_anneal": True}}},
    ],
    # C1. Shuffled-time control --- temporal causality test
    "C1_shuffled_time": [
        {"id": "true",     "patch": {"sonostate": {"shuffle_pairs": False}}},
        {"id": "shuffled", "patch": {"sonostate": {"shuffle_pairs": True}}},
    ],
}


def deep_merge(dst: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base SonoState training YAML.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--exp-root",
                   default="/mnt/vast/exp/brucexu/EchoJEPA/experiments_2026",
                   help="Where each cell writes checkpoints.")
    p.add_argument("--data-csv",
                   default="/mnt/vast/data/brucexu/echonet/pretrain_annotations.csv",
                   help="Annotation CSV for SonoState training (overrides data.datasets).")
    p.add_argument("--anneal-ckpt",
                   default="/mnt/vast/checkpoints/brucexu/vjepa2/vitl.pt",
                   help="Pretrained V-JEPA-2 checkpoint (overrides optimization.anneal_ckpt).")
    args = p.parse_args()

    with open(args.base) as f:
        base = yaml.safe_load(f)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sweep_name, cells in SWEEPS.items():
        sweep_dir = out / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            cfg = copy.deepcopy(base)
            deep_merge(cfg, cell["patch"])
            # Rewrite data + checkpoint paths to cluster-local locations
            cfg.setdefault("data", {})["datasets"] = [args.data_csv]
            cfg["data"].setdefault("datasets_weights", [1.0])
            cfg.setdefault("optimization", {})["anneal_ckpt"] = args.anneal_ckpt
            # Each cell writes to its own folder
            cfg["folder"] = f"{args.exp_root}/{sweep_name}/{cell['id']}"
            # Pin a stable seed per cell so results are reproducible
            cfg.setdefault("meta", {})["seed"] = 234 + abs(hash(cell["id"])) % 1000
            cell_path = sweep_dir / f"{cell['id']}.yaml"
            with open(cell_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            manifest.append({
                "sweep": sweep_name,
                "cell": cell["id"],
                "config": str(cell_path),
                "folder": cfg["folder"],
            })

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} configs across {len(SWEEPS)} sweeps.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
