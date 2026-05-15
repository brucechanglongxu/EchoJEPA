"""Geometry of the latent state space (Leg L4).

Three measurements:
  (1) Intrinsic dimension via TwoNN (Facco et al., 2017).
  (2) View-conditioned silhouette score (requires a `view` column in CSV).
  (3) Loop closure: per-video maximum great-circle distance from start state
      vs. expected diameter; closer to 0 -> tighter closed loop.

Outputs metrics.json and a small bar chart.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

from experiments.analysis._common import (
    resolve_video_path,
    extract_video_states,
    load_sonostate,
    setup_logging,
)


def twonn_id(X: np.ndarray) -> float:
    """TwoNN intrinsic-dimension estimator. X: (N, d)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    d, _ = tree.query(X, k=3)  # self + 2 neighbours
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = mu[mu > 1.0]
    if len(mu) < 50:
        return float("nan")
    F = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    x = np.log(np.sort(mu))
    y = -np.log(1 - F)
    # Linear fit through origin: dim = slope
    return float(np.sum(x * y) / np.sum(x * x))


def loop_closure(z: np.ndarray) -> float:
    """Mean cosine distance between first and last state, normalized by
    the maximum cosine distance traversed."""
    if z.shape[0] < 4:
        return float("nan")
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    closure = 1.0 - float(zn[0] @ zn[-1])
    diameter = 1.0 - float(np.min(zn @ zn.T))
    if diameter < 1e-6:
        return float("nan")
    return closure / diameter


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True,
                   help="CSV with columns: path[, view]")
    p.add_argument("--video-root", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max-videos", type=int, default=1000)
    p.add_argument("--state-dim", type=int, default=256)
    p.add_argument("--min-states-per-video", type=int, default=2)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = load_sonostate(args.checkpoint, device, state_dim=args.state_dim)

    try:
        df = pd.read_csv(args.csv)
    except Exception:
        df = pd.read_csv(args.csv, sep=r"\s+", header=None,
                         names=["path", "view"])
    if "path" not in df.columns and "FileName" in df.columns:
        df = df.rename(columns={"FileName": "path"})
    df = df.head(args.max_videos)

    pooled = []                         # all states pooled (for ID)
    means = []                          # per-video mean state
    views = []                          # per-video view label
    closures = []
    for _, row in df.iterrows():
        path = resolve_video_path(row["path"], args.video_root)
        if not os.path.isfile(path):
            continue
        try:
            z, _ = extract_video_states(bundle, path, device)
        except Exception:
            continue
        if z.shape[0] < args.min_states_per_video:
            continue
        pooled.append(z)
        means.append(z.mean(axis=0))
        if "view" in row:
            views.append(row["view"])
        closures.append(loop_closure(z))

    Z = np.concatenate(pooled, axis=0) if pooled else np.empty((0, args.state_dim))
    means = np.stack(means) if means else np.empty((0, args.state_dim))

    metrics = {
        "n_videos": len(closures),
        "n_states": int(Z.shape[0]),
        "intrinsic_dim_twonn": twonn_id(Z) if Z.shape[0] >= 100 else None,
        "loop_closure_mean": float(np.nanmean(closures)) if closures else None,
        "loop_closure_median": float(np.nanmedian(closures)) if closures else None,
    }
    if views and len(set(views)) > 1 and means.shape[0] == len(views):
        # silhouette on per-video means
        try:
            metrics["view_silhouette"] = float(silhouette_score(means, views, metric="cosine"))
        except Exception as e:
            metrics["view_silhouette"] = None
            print(f"silhouette failed: {e}")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
