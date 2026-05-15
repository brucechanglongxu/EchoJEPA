"""Heart-rate decoding from latent angular speed (Leg L3).

If z_t orbits a closed cardiac-cycle loop, the angular speed of the
trajectory in 2D PCA should be proportional to heart rate (HR). EchoNet-
Dynamic provides per-video FrameHeight/Width and FPS metadata; the period
of the loop (in clip-steps) times the clip stride (in real seconds) gives
an estimate of beat duration, hence HR.

Ground-truth HR is approximated from the time between consecutive
ED frames in the *same* video where available, otherwise from a
manually-curated `hr_csv` (optional).

Outputs:
  metrics.json: pearson_r, spearman_r, mae_bpm, n
  figure: hr_correlation.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

from experiments.analysis._common import (
    resolve_video_path,
    extract_video_states,
    load_sonostate,
    setup_logging,
)


def estimate_period(z_traj: np.ndarray, pca: PCA) -> float | None:
    """Return the dominant period (in clip-steps) of the latent loop.

    We project to PCA space and use peak-finding on PC1.
    """
    if z_traj.shape[0] < 8:
        return None
    proj = pca.transform(z_traj)[:, 0]
    proj = (proj - np.mean(proj)) / (np.std(proj) + 1e-8)
    # Require non-trivial peaks (prominence) and at least one full cycle.
    peaks, _ = find_peaks(proj, distance=4, prominence=0.3)
    if len(peaks) < 2:
        return None
    diffs = np.diff(peaks)
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--filelist-csv", required=True)
    p.add_argument("--vt-csv", default=None,
                   help="Optional VolumeTracings.csv. If provided, the ED/ES "
                        "frame interval is used as a coarse bpm ground-truth "
                        "(period ≈ 2.5 * |ED-ES| frames, since systole is "
                        "typically ~0.4 of the cardiac cycle).")
    p.add_argument("--video-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="VAL")
    p.add_argument("--max-videos", type=int, default=1000)
    p.add_argument("--state-dim", type=int, default=256)
    p.add_argument("--fps-target", type=int, default=8)
    p.add_argument("--stride-frames", type=int, default=4)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = load_sonostate(args.checkpoint, device, state_dim=args.state_dim)

    fl = pd.read_csv(args.filelist_csv)
    fl = fl[fl["Split"].str.upper() == args.split.upper()]

    # Build a bpm ground-truth lookup from VolumeTracings (ED/ES frames).
    bpm_gt: dict[str, float] = {}
    if args.vt_csv is not None and os.path.isfile(args.vt_csv):
        vt = pd.read_csv(args.vt_csv)
        # Normalize FileName: VT typically has '.avi', FileList does not.
        vt["FileName"] = vt["FileName"].astype(str).str.replace(
            r"\.avi$", "", regex=True
        )
        for fn, sub in vt.groupby("FileName"):
            frames = sorted(sub["Frame"].unique().tolist())
            if len(frames) != 2:
                continue
            meta = fl[fl["FileName"] == fn]
            if meta.empty:
                continue
            video_fps = float(meta.iloc[0].get("FPS", 50.0))
            half = abs(frames[1] - frames[0]) / max(1.0, video_fps)
            # Systolic interval is ~40% of full cycle in adults; rescale
            # to a period estimate, then convert to bpm.
            full_cycle_sec = half / 0.4
            if full_cycle_sec <= 0.2:
                continue
            bpm_gt[fn] = 60.0 / full_cycle_sec
        print(f"[hr-gt] derived bpm ground-truth for {len(bpm_gt)} videos")

    if not bpm_gt and "HR" not in fl.columns and "HeartRate" not in fl.columns:
        print("[warn] No bpm ground truth available; reporting predictions only.")
    fl = fl.head(args.max_videos)

    # Pass 1: gather all states to fit a single PCA
    cache: list[tuple[str, np.ndarray]] = []
    for _, row in fl.iterrows():
        path = resolve_video_path(row["FileName"], args.video_root)
        if not os.path.isfile(path):
            continue
        try:
            z, _ = extract_video_states(
                bundle, path, device,
                fps_target=args.fps_target,
                stride_frames=args.stride_frames,
            )
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        if z.shape[0] >= 6:
            cache.append((row["FileName"], z))
    if not cache:
        raise SystemExit("No usable videos.")
    Z_all = np.concatenate([z for _, z in cache], axis=0)
    pca = PCA(n_components=2).fit(Z_all)

    # Pass 2: estimate per-video period -> HR (bpm)
    rows = []
    for fn, z in cache:
        period_steps = estimate_period(z, pca)
        if period_steps is None:
            continue
        # Each step in z_t corresponds to args.stride_frames / video_fps seconds.
        # We don't have per-video fps loaded from decord here; assume the column.
        meta = fl[fl["FileName"] == fn]
        if meta.empty:
            continue
        video_fps = float(meta.iloc[0].get("FPS", 50.0))
        seconds_per_step = args.stride_frames / max(1.0, video_fps)
        period_sec = period_steps * seconds_per_step
        if period_sec <= 0.05:
            continue
        bpm_pred = 60.0 / period_sec

        bpm_true = bpm_gt.get(fn)
        if bpm_true is None:
            for col in ("HR", "HeartRate"):
                if col in meta.columns and pd.notna(meta.iloc[0][col]):
                    bpm_true = float(meta.iloc[0][col])
                    break
        rows.append({"FileName": fn, "bpm_pred": bpm_pred, "bpm_true": bpm_true})

    df = pd.DataFrame(rows)
    df.to_csv(out / "hr_predictions.csv", index=False)
    paired = df.dropna(subset=["bpm_true"])
    if len(paired) >= 10:
        r, _ = pearsonr(paired["bpm_pred"], paired["bpm_true"])
        rho, _ = spearmanr(paired["bpm_pred"], paired["bpm_true"])
        mae = float(np.mean(np.abs(paired["bpm_pred"] - paired["bpm_true"])))
    else:
        r, rho, mae = None, None, None

    metrics = {
        "n_total": len(df),
        "n_paired_with_gt": int(paired.shape[0]),
        "pearson_r": r,
        "spearman_r": rho,
        "mae_bpm": mae,
        "median_pred_bpm": float(df["bpm_pred"].median()),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    if not paired.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4.5, 4.5))
            plt.scatter(paired["bpm_true"], paired["bpm_pred"], alpha=0.5, s=10)
            lim = (30, 200)
            plt.plot(lim, lim, "k--", lw=0.5)
            plt.xlabel("ground-truth HR (bpm)")
            plt.ylabel("decoded HR (bpm)")
            plt.title(f"r={r:.2f} | MAE={mae:.1f} bpm | n={len(paired)}")
            plt.tight_layout()
            plt.savefig(out / "hr_correlation.pdf")
        except Exception as e:
            print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
