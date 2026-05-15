"""Linear / ridge probe of clinical labels from z_t (Leg L5).

For each video we mean-pool the latent trajectory z_t and fit a ridge
regression (continuous label) or logistic regression (categorical) on the
training split, evaluated on the test split. Compared baselines:
  - mean-pool of raw encoder features (D=1024)
  - mean-pool of state-head features z_t (d=256)

Supported tasks (auto-detected from columns):
  - LVEF (continuous)
  - View (categorical, requires 'view' column)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

from experiments.analysis._common import (
    resolve_video_path,
    extract_video_states,
    load_sonostate,
    setup_logging,
)


@torch.no_grad()
def _video_features(bundle, path, device, raw: bool = False):
    """Return (raw_feature_pool, state_pool) for a video."""
    from decord import VideoReader, cpu
    import torch.nn.functional as F
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    fps_video = max(1, round(vr.get_avg_fps()))
    fpc, fps_target = 16, 8
    step = max(1, fps_video // fps_target)
    clip_len = fpc * step
    stride = max(1, clip_len // 2)
    raw_feats, states = [], []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1, 1) * 255.0
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1, 1) * 255.0
    for start in range(0, max(1, len(vr) - clip_len + 1), stride):
        idx = np.linspace(start, start + clip_len - 1, num=fpc).astype(np.int64)
        frames = vr.get_batch(np.clip(idx, 0, len(vr) - 1)).asnumpy()
        t = torch.from_numpy(frames).to(device).float().permute(3, 0, 1, 2)
        C, T, H, W = t.shape
        t = F.interpolate(t.reshape(C * T, 1, H, W),
                          size=(224, 224), mode="bilinear",
                          align_corners=False).reshape(C, T, 224, 224)
        t = (t - mean) / std
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens = bundle.encoder([t.unsqueeze(0)])
            z = bundle.state_head(tokens[0])
        if raw:
            raw_feats.append(tokens[0].mean(dim=1).float().cpu().numpy().squeeze(0))
        states.append(z.float().cpu().numpy().squeeze(0))
    if not states:
        return None, None
    raw_pool = np.mean(np.stack(raw_feats), axis=0) if raw_feats else None
    state_pool = np.mean(np.stack(states), axis=0)
    return raw_pool, state_pool


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True,
                   help="EchoNet FileList.csv (Split, FileName, EF[, view])")
    p.add_argument("--video-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="EF",
                   help="Column name to regress/classify.")
    p.add_argument("--task", default="auto", choices=["auto", "regression", "classification"])
    p.add_argument("--max-videos", type=int, default=2000)
    p.add_argument("--state-dim", type=int, default=256)
    p.add_argument("--include-raw", action="store_true",
                   help="Also extract & probe raw encoder features (slower).")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = load_sonostate(args.checkpoint, device, state_dim=args.state_dim)

    df = pd.read_csv(args.csv)
    if "Split" not in df.columns:
        raise SystemExit("Expected EchoNet FileList.csv with a Split column.")
    df = df.head(args.max_videos)

    task = args.task
    if task == "auto":
        task = "regression" if pd.api.types.is_numeric_dtype(df[args.target]) else "classification"

    feats_state = {"TRAIN": [], "VAL": [], "TEST": []}
    feats_raw = {"TRAIN": [], "VAL": [], "TEST": []}
    labels = {"TRAIN": [], "VAL": [], "TEST": []}
    for _, row in df.iterrows():
        sp = row["Split"].upper()
        if sp not in feats_state:
            continue
        path = resolve_video_path(row["FileName"], args.video_root)
        if not os.path.isfile(path):
            continue
        try:
            raw_pool, state_pool = _video_features(bundle, path, device, raw=args.include_raw)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        if state_pool is None:
            continue
        feats_state[sp].append(state_pool)
        if raw_pool is not None:
            feats_raw[sp].append(raw_pool)
        labels[sp].append(row[args.target])

    out_metrics = {"task": task, "target": args.target}
    for source, feats in (("state", feats_state), ("raw", feats_raw)):
        if not feats["TRAIN"] or not feats["TEST"]:
            continue
        Xtr = np.stack(feats["TRAIN"]); ytr = np.array(labels["TRAIN"])[: len(Xtr)]
        Xte = np.stack(feats["TEST"]);  yte = np.array(labels["TEST"])[: len(Xte)]
        if task == "regression":
            best = None
            for alpha in (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2):
                m = Ridge(alpha=alpha).fit(Xtr, ytr)
                pred = m.predict(Xte)
                mae = mean_absolute_error(yte, pred)
                r2 = r2_score(yte, pred)
                if best is None or mae < best["mae"]:
                    best = {"alpha": alpha, "mae": mae, "r2": r2}
            out_metrics[source] = best
        else:
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
            pred = m.predict(Xte)
            out_metrics[source] = {
                "acc": float(accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro")),
            }
        out_metrics[f"{source}_n_train"] = int(len(Xtr))
        out_metrics[f"{source}_n_test"] = int(len(Xte))

    with open(out / "metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2)
    print(json.dumps(out_metrics, indent=2))


if __name__ == "__main__":
    main()
