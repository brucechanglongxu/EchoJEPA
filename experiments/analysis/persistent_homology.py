"""Persistent homology (H_0, H_1) of the pooled latent state cloud.

Hardens the "closed manifold" claim against the criticism that the
visible loop in 2D PCA is a linear artifact. If the state space truly
forms a 1-cycle (S^1-like topology) we expect:

  - a single H_1 feature with persistence much larger than the rest,
  - a dominance ratio (top H_1 persistence / second-largest) >> 1,
  - H_0 collapsing to a single connected component well before the
    dominant H_1 feature is born.

We pool states from held-out EchoNet-Dynamic videos, optionally
project to the d_pca leading PCs (to keep PH tractable in ~3-D where
the manifold actually lives), subsample to N_sub points, and run
``ripser`` with ``maxdim=1``.

Outputs:
  metrics.json with summary statistics
  ph_diagram.png with the H_0/H_1 persistence diagram (best-effort)
  Z_pooled.npy cache for re-runs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.analysis._common import (
    extract_video_states,
    load_sonostate,
    resolve_video_path,
    setup_logging,
)


def _pool_states(
    bundle, df, video_root: str, device, max_videos: int, min_per_video: int
) -> np.ndarray:
    pooled = []
    for _, row in df.head(max_videos).iterrows():
        path = resolve_video_path(row["path"], video_root)
        if not os.path.isfile(path):
            continue
        try:
            z, _ = extract_video_states(bundle, path, device)
        except Exception:
            continue
        if z.shape[0] < min_per_video:
            continue
        pooled.append(z)
    if not pooled:
        return np.empty((0, 0))
    return np.concatenate(pooled, axis=0)


def _persistence_summary(diagrams: list[np.ndarray]) -> dict:
    """Compute persistence-bar statistics for H_0 and H_1."""
    summary: dict = {}
    for k, dgm in enumerate(diagrams):
        if dgm.size == 0:
            summary[f"H{k}_n_features"] = 0
            continue
        # Drop the H_0 essential bar (infinite death) for ratio stats.
        finite = dgm[np.isfinite(dgm[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0] if finite.size else np.array([])
        order = np.argsort(lifetimes)[::-1] if lifetimes.size else np.array([], dtype=int)
        summary[f"H{k}_n_features"] = int(dgm.shape[0])
        summary[f"H{k}_n_finite"] = int(finite.shape[0])
        summary[f"H{k}_top5_persistence"] = [
            float(lifetimes[i]) for i in order[:5]
        ]
        if lifetimes.size >= 2:
            top, second = float(lifetimes[order[0]]), float(lifetimes[order[1]])
            summary[f"H{k}_dominance_ratio"] = top / max(second, 1e-12)
        elif lifetimes.size == 1:
            summary[f"H{k}_dominance_ratio"] = float("inf")
        else:
            summary[f"H{k}_dominance_ratio"] = None
    return summary


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--video-root", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max-videos", type=int, default=1000)
    p.add_argument("--min-states-per-video", type=int, default=2)
    p.add_argument("--state-dim", type=int, default=256)
    p.add_argument("--d-pca", type=int, default=8,
                   help="Project to top-d PCs before PH (0 = no PCA).")
    p.add_argument("--n-sub", type=int, default=600,
                   help="Subsample size for ripser (PH is O(n^3) worst case).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="",
                   help="Optional path to cached Z .npy (skip extraction).")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cache and os.path.isfile(args.cache):
        Z = np.load(args.cache)
        print(f"[cache] loaded Z {Z.shape} from {args.cache}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bundle = load_sonostate(args.checkpoint, device, state_dim=args.state_dim)
        try:
            df = pd.read_csv(args.csv)
        except Exception:
            df = pd.read_csv(args.csv, sep=r"\s+", header=None,
                             names=["path", "view"])
        if "path" not in df.columns and "FileName" in df.columns:
            df = df.rename(columns={"FileName": "path"})
        Z = _pool_states(
            bundle, df, args.video_root, device,
            args.max_videos, args.min_states_per_video,
        )
        np.save(out / "Z_pooled.npy", Z)
        print(f"[extract] Z {Z.shape} -> {out / 'Z_pooled.npy'}")

    if Z.shape[0] < 50:
        raise RuntimeError(f"Too few states: {Z.shape}")

    # Normalise to the unit sphere (states are already L2-normalised at
    # the head; redo for safety in case of cached older runs).
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    # Optional PCA projection to keep ripser fast and to test that the
    # topology lives in the low-d manifold rather than in the ambient
    # head dimension.
    Z_proj = Z
    if args.d_pca > 0 and args.d_pca < Z.shape[1]:
        Zc = Z - Z.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        Z_proj = Zc @ Vt[: args.d_pca].T
        # Re-normalise so distances are comparable across cells.
        Z_proj = Z_proj / np.maximum(np.linalg.norm(Z_proj, axis=1, keepdims=True), 1e-12)

    # Subsample for ripser.
    rng = np.random.default_rng(args.seed)
    n_sub = min(args.n_sub, Z_proj.shape[0])
    idx = rng.choice(Z_proj.shape[0], size=n_sub, replace=False)
    Xs = Z_proj[idx]

    from ripser import ripser
    res = ripser(Xs, maxdim=1)
    dgms = res["dgms"]  # list: [H0, H1]

    summary = _persistence_summary(dgms)
    summary.update({
        "n_states_total": int(Z.shape[0]),
        "n_states_used": int(Xs.shape[0]),
        "d_ambient": int(Z.shape[1]),
        "d_pca": int(args.d_pca) if args.d_pca > 0 else None,
        "seed": int(args.seed),
    })

    # Persist diagrams as .npy for plotting / external inspection.
    for k, dgm in enumerate(dgms):
        np.save(out / f"H{k}_diagram.npy", dgm)

    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Best-effort PH diagram figure.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        for k, dgm in enumerate(dgms):
            if dgm.size == 0:
                continue
            finite = dgm[np.isfinite(dgm[:, 1])]
            if finite.size:
                ax.scatter(finite[:, 0], finite[:, 1], s=14,
                           label=f"H{k}", alpha=0.7)
        lo, hi = 0.0, float(np.nanmax([
            np.nanmax(d[np.isfinite(d[:, 1])][:, 1]) if d[np.isfinite(d[:, 1])].size else 0.0
            for d in dgms
        ]) * 1.05 + 1e-6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        ax.set_title("Persistence diagram")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(out / "ph_diagram.pdf")
        fig.savefig(out / "ph_diagram.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
