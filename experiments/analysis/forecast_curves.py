"""Forecast quality (Leg L1).

Computes single- and multi-step forecast error of the SonoState transition
operator vs. a persistence baseline, on EchoNet-Dynamic test split.

Outputs:
    metrics.json: {h: {forecast_l1, persistence_l1, forecast_cos,
                       persistence_cos, n}, ...}
    figure: forecast_curves.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.analysis._common import (
    extract_video_states,
    load_sonostate,
    resolve_video_path,
    setup_logging,
)


def _read_split_csv(csv_path: str, split: str | None) -> pd.DataFrame:
    """EchoNet's FileList.csv has a Split column; the SonoState training CSVs
    are space-delimited (path, label). Accept either."""
    try:
        df = pd.read_csv(csv_path)
        if split and "Split" in df.columns:
            df = df[df["Split"].str.upper() == split.upper()]
        if "FileName" in df.columns:
            df = df.rename(columns={"FileName": "path"})
        return df
    except Exception:
        df = pd.read_csv(csv_path, sep=r"\s+", header=None, names=["path", "label"])
        return df


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True, help="Eval CSV (FileList.csv or SonoState train csv).")
    p.add_argument("--video-root", default="", help="Prefix for relative paths.")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="TEST")
    p.add_argument("--max-videos", type=int, default=200)
    p.add_argument("--max-horizon", type=int, default=10)
    p.add_argument("--state-dim", type=int, default=256)
    p.add_argument("--transition-hidden-dim", type=int, default=512)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bundle = load_sonostate(
        args.checkpoint, device,
        state_dim=args.state_dim,
        transition_hidden_dim=args.transition_hidden_dim,
    )

    df = _read_split_csv(args.csv, args.split)
    if "path" not in df.columns:
        raise SystemExit(f"Cannot find 'path' column in {args.csv}")

    horizons = list(range(1, args.max_horizon + 1))
    f_l1 = {h: [] for h in horizons}
    f_cos = {h: [] for h in horizons}
    p_l1 = {h: [] for h in horizons}
    p_cos = {h: [] for h in horizons}

    n_used = 0
    for _, row in df.iterrows():
        if n_used >= args.max_videos:
            break
        rel = row["path"]
        path = rel if os.path.isabs(rel) else resolve_video_path(rel, args.video_root)
        if not os.path.isfile(path):
            continue
        try:
            z, _ = extract_video_states(bundle, path, device)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        if z.shape[0] < 2:
            continue
        z_t = torch.from_numpy(z).to(device)

        # multi-step rollout from each starting clip
        for t in range(z_t.size(0) - 1):
            z0 = z_t[t]
            z_roll = z0
            for h in horizons:
                if t + h >= z_t.size(0):
                    break
                z_roll = bundle.transition(z_roll.unsqueeze(0)).squeeze(0)
                z_true = z_t[t + h]
                f_l1[h].append(F.l1_loss(z_roll, z_true).item())
                f_cos[h].append(1.0 - F.cosine_similarity(
                    z_roll.unsqueeze(0), z_true.unsqueeze(0)).item())
                p_l1[h].append(F.l1_loss(z0, z_true).item())
                p_cos[h].append(1.0 - F.cosine_similarity(
                    z0.unsqueeze(0), z_true.unsqueeze(0)).item())
        n_used += 1

    metrics = {}
    for h in horizons:
        if not f_l1[h]:
            continue
        metrics[h] = {
            "forecast_l1": float(np.mean(f_l1[h])),
            "forecast_cos": float(np.mean(f_cos[h])),
            "persistence_l1": float(np.mean(p_l1[h])),
            "persistence_cos": float(np.mean(p_cos[h])),
            "n": len(f_l1[h]),
        }
    summary = {"n_videos": n_used, "horizons": metrics}
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # quick figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hs = sorted(metrics.keys())
        plt.figure(figsize=(5.5, 4))
        plt.plot(hs, [metrics[h]["persistence_l1"] for h in hs], "o--", label="persistence")
        plt.plot(hs, [metrics[h]["forecast_l1"] for h in hs], "s-", label="SonoState")
        plt.xlabel("horizon h (clips)")
        plt.ylabel(r"L1$\bigl(z_{t+h},\hat z_{t+h}\bigr)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "forecast_curves.pdf")
        print(f"wrote {out/'forecast_curves.pdf'}")
    except Exception as e:
        print(f"plotting failed: {e}")
    print(f"wrote {out/'metrics.json'}")


if __name__ == "__main__":
    main()
