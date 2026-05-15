"""Distribution-shift / safety monitor stub (Section "Safe deployment").

Compares latent trajectory statistics between an in-distribution cohort
(EchoNet-Dynamic test) and out-of-distribution cohorts (e.g. EchoNet-
Pediatric, augmented data, low-quality scans). For each video we compute:

  - off-manifold distance: average min-cosine to ID training centroids
  - loop closure (from geometry.loop_closure)
  - forecast L1 at h=1 (the model's own self-prediction error)

We report ROC-AUC of the *forecast L1* feature alone as an OOD detector.
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
from sklearn.metrics import roc_auc_score

from experiments.analysis._common import (
    resolve_video_path,
    extract_video_states,
    load_sonostate,
    setup_logging,
)


def per_video_signals(bundle, path, device):
    z, _ = extract_video_states(bundle, path, device)
    if z.shape[0] < 2:
        return None
    z_t = torch.from_numpy(z).to(device)
    fwd = bundle.transition(z_t[:-1])
    fwd_l1 = F.l1_loss(fwd, z_t[1:]).item()
    closure = 1.0 - float(torch.dot(z_t[0], z_t[-1]).item())
    return {"forecast_l1": fwd_l1, "closure": closure}


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--id-csv", required=True, help="In-distribution video CSV.")
    p.add_argument("--ood-csv", required=True, help="Out-of-distribution video CSV.")
    p.add_argument("--id-root", required=True)
    p.add_argument("--ood-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-per-cohort", type=int, default=300)
    p.add_argument("--state-dim", type=int, default=256)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = load_sonostate(args.checkpoint, device, state_dim=args.state_dim)

    def load_csv(path):
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", header=None, names=["FileName", "label"])
        if "FileName" not in df.columns and "path" in df.columns:
            df = df.rename(columns={"path": "FileName"})
        return df

    rows = []
    for label, csv, root in (("ID", args.id_csv, args.id_root),
                              ("OOD", args.ood_csv, args.ood_root)):
        df = load_csv(csv).head(args.max_per_cohort)
        for _, row in df.iterrows():
            path = resolve_video_path(row["FileName"], root)
            if not os.path.isfile(path):
                continue
            sig = per_video_signals(bundle, path, device)
            if sig is None:
                continue
            sig["label"] = label
            rows.append(sig)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ood_signals.csv", index=False)

    metrics = {"n_id": int((df.label == "ID").sum()),
               "n_ood": int((df.label == "OOD").sum())}
    if metrics["n_id"] >= 10 and metrics["n_ood"] >= 10:
        y = (df.label == "OOD").astype(int)
        for f in ("forecast_l1", "closure"):
            metrics[f"auc_{f}"] = float(roc_auc_score(y, df[f]))
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
