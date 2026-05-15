"""Phase recovery from EchoNet-Dynamic ED/ES annotations (Leg L2).

EchoNet-Dynamic ships VolumeTracings.csv with the *frame number* of two
reference cardiac-cycle keypoints per video: end-systole (ES) and
end-diastole (ED). Both are unique points on the cycle, ED at maximum
chamber volume and ES at minimum.

Hypothesis: if z_t encodes cardiac phase, then:
  (a) z(ED_video) and z(ES_video) lie at consistent angular positions
      on the latent loop (across patients);
  (b) the angular distance between any pair (ED, ES) within a video is
      close to pi (they are anti-podal on the cycle);
  (c) a logistic regression on z can decode {ED, ES} with high accuracy.

This script computes (a) circular variance of ED and ES angles, (b) mean
angular separation, and (c) ED-vs-ES classification accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from experiments.analysis._common import (
    resolve_video_path,
    load_sonostate,
    setup_logging,
)


def _read_videos(filelist_csv: str, vt_csv: str, split: str) -> dict[str, list[int]]:
    """Return {video_path_basename: [ED_frame, ES_frame]} for the requested split.

    EchoNet-Dynamic VolumeTracings.csv has columns: FileName, X1, Y1, X2, Y2, Frame.
    Only two unique Frame values per video (ED, ES).  By convention the first
    occurring Frame is ED (larger volume); we determine ED/ES by counting
    tracing points (more tracings -> ED), but we also keep both as a
    binary label to be agnostic.
    """
    fl = pd.read_csv(filelist_csv)
    fl = fl[fl["Split"].str.upper() == split.upper()]
    valid = set(fl["FileName"].tolist())
    vt = pd.read_csv(vt_csv)
    # VT FileName has '.avi'; FileList does not. Normalize so the join works.
    vt["FileName"] = vt["FileName"].astype(str).str.replace(
        r"\.avi$", "", regex=True
    )
    grouped: dict[str, list[int]] = {}
    for fn, sub in vt.groupby("FileName"):
        if fn not in valid:
            continue
        # Two distinct frame indices
        frames = sorted(sub["Frame"].unique().tolist())
        if len(frames) != 2:
            continue
        # Use number of tracing points as proxy: ED has more chord lines
        # because the chamber is larger. (Heuristic: the frame with more rows
        # in VolumeTracings is ED.)
        counts = sub.groupby("Frame").size().to_dict()
        f0, f1 = frames
        if counts[f0] >= counts[f1]:
            ed, es = f0, f1
        else:
            ed, es = f1, f0
        grouped[fn] = [int(ed), int(es)]
    return grouped


def _circular_var(angles: np.ndarray) -> float:
    """Circular variance in [0, 1]; 0 = perfectly concentrated."""
    if len(angles) == 0:
        return float("nan")
    R = np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
    return float(1.0 - R)


@torch.no_grad()
def _state_at_frame(bundle, video_path: str, frame_idx: int, device,
                    crop_size: int = 224, fpc: int = 16, fps_target: int = 8):
    """Extract z for a clip *centered* on the requested frame."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    fps_video = max(1, round(vr.get_avg_fps()))
    step = max(1, fps_video // fps_target)
    clip_len = fpc * step
    start = max(0, frame_idx - clip_len // 2)
    start = min(start, len(vr) - clip_len)
    if start < 0:
        return None
    indices = np.linspace(start, start + clip_len - 1, num=fpc).astype(np.int64)
    indices = np.clip(indices, 0, len(vr) - 1)
    frames = vr.get_batch(indices).asnumpy()
    t = torch.from_numpy(frames).to(device).float().permute(3, 0, 1, 2)
    C, T, H, W = t.shape
    t = F.interpolate(
        t.reshape(C * T, 1, H, W), size=(crop_size, crop_size),
        mode="bilinear", align_corners=False,
    ).reshape(C, T, crop_size, crop_size)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1, 1) * 255.0
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1, 1) * 255.0
    t = (t - mean) / std
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        tokens = bundle.encoder([t.unsqueeze(0)])
        z = bundle.state_head(tokens[0])
    return z.detach().float().cpu().numpy().squeeze(0)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--filelist-csv", required=True,
                   help="Path to EchoNet-Dynamic FileList.csv")
    p.add_argument("--vt-csv", required=True,
                   help="Path to EchoNet-Dynamic VolumeTracings.csv")
    p.add_argument("--video-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="VAL")
    p.add_argument("--max-videos", type=int, default=300)
    p.add_argument("--state-dim", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = load_sonostate(args.checkpoint, device, state_dim=args.state_dim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = _read_videos(args.filelist_csv, args.vt_csv, args.split)
    items = list(pairs.items())[: args.max_videos]
    print(f"Decoding phase from {len(items)} videos (split={args.split})...")

    Z, y = [], []  # y: 0=ED, 1=ES
    Z_ed, Z_es = [], []
    for fn, (ed_f, es_f) in items:
        path = resolve_video_path(fn, args.video_root)
        if not os.path.isfile(path):
            continue
        z_ed = _state_at_frame(bundle, path, ed_f, device)
        z_es = _state_at_frame(bundle, path, es_f, device)
        if z_ed is None or z_es is None:
            continue
        Z.append(z_ed); y.append(0); Z_ed.append(z_ed)
        Z.append(z_es); y.append(1); Z_es.append(z_es)

    Z = np.stack(Z); y = np.array(y)
    Z_ed = np.stack(Z_ed); Z_es = np.stack(Z_es)
    print(f"got {len(Z_ed)} ED and {len(Z_es)} ES states")

    # (a) circular variance of ED and ES angles in 2D PCA
    pca = PCA(n_components=2).fit(Z)
    ang_ed = np.arctan2(*pca.transform(Z_ed)[:, ::-1].T)
    ang_es = np.arctan2(*pca.transform(Z_es)[:, ::-1].T)
    cv_ed = _circular_var(ang_ed)
    cv_es = _circular_var(ang_es)

    # (b) mean angular separation between ED and ES (paired)
    ang_diff = np.abs(np.angle(np.exp(1j * (ang_ed - ang_es))))
    mean_sep = float(np.mean(ang_diff))

    # (c) cross-validated ED-vs-ES classification accuracy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(Z, y):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y[tr])
        accs.append(float(clf.score(Z[te], y[te])))
    cls_acc = float(np.mean(accs))

    # baseline: same with shuffled labels
    rng = np.random.default_rng(0)
    accs_shuf = []
    for tr, te in skf.split(Z, y):
        y_shuf = rng.permutation(y[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y_shuf)
        accs_shuf.append(float(clf.score(Z[te], y[te])))
    cls_shuf = float(np.mean(accs_shuf))

    metrics = {
        "n_pairs": len(Z_ed),
        "circular_var_ED": cv_ed,
        "circular_var_ES": cv_es,
        "mean_angular_separation_rad": mean_sep,
        "mean_angular_separation_deg": float(np.degrees(mean_sep)),
        "ed_vs_es_classification_acc": cls_acc,
        "ed_vs_es_classification_acc_shuffled": cls_shuf,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        for ang, lab, c in [(ang_ed, "ED", "tab:red"), (ang_es, "ES", "tab:blue")]:
            ax.hist(np.degrees(ang), bins=24, alpha=0.6, label=lab, color=c)
        ax.set_xlabel("angle on PC1-PC2 plane (deg)")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title(
            f"circ-var ED={cv_ed:.2f}, ES={cv_es:.2f}, sep={np.degrees(mean_sep):.0f}°"
        )
        ax = axes[1]
        ax.bar(["chance", "shuffled", "SonoState"], [0.5, cls_shuf, cls_acc],
               color=["#bbb", "#bbb", "tab:green"])
        ax.set_ylabel("ED vs ES classification accuracy")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out / "phase_recovery.pdf")
    except Exception as e:
        print(f"plot failed: {e}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
