"""Cycle-consistency control (Control C3).

Train an *inverse* transition g_phi alongside the forward f_theta, then
measure post-hoc:
  - One-step round-trip error: ||g(f(z)) - z||
  - h-step round-trip error:   ||g^h(f^h(z)) - z||

This is a sanity check: if z lives on a low-dim cycle, a frozen f should
admit a learnable inverse with bounded round-trip error. If the latent
space is collapsed or chaotic, error grows with h.

Note: g is fit *post-hoc* on a holdout set; we never modify the
SonoState checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.analysis._common import (
    resolve_video_path,
    extract_video_states,
    load_sonostate,
    setup_logging,
)


class InverseTransition(nn.Module):
    def __init__(self, state_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.log_scale = nn.Parameter(torch.tensor(-4.6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = self.log_scale.exp()
        return F.normalize(z + s * self.net(z), dim=-1)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-videos", type=int, default=300)
    p.add_argument("--state-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = load_sonostate(args.checkpoint, device, state_dim=args.state_dim)

    df = pd.read_csv(args.csv) if args.csv.endswith(".csv") else pd.read_csv(
        args.csv, sep=r"\s+", header=None, names=["FileName", "label"])
    if "FileName" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "FileName"})

    pairs = []  # (z_t, z_{t+1})
    for _, row in df.head(args.max_videos).iterrows():
        path = resolve_video_path(row["FileName"], args.video_root)
        if not os.path.isfile(path):
            continue
        try:
            z, _ = extract_video_states(bundle, path, device)
        except Exception:
            continue
        for t in range(z.shape[0] - 1):
            pairs.append((z[t], z[t + 1]))
    if not pairs:
        raise SystemExit("No pairs.")
    Z_t = torch.tensor(np.stack([p[0] for p in pairs])).to(device)
    Z_n = torch.tensor(np.stack([p[1] for p in pairs])).to(device)
    n = Z_t.size(0); n_train = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[:n_train], perm[n_train:]

    g = InverseTransition(state_dim=args.state_dim).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=3e-4)
    for epoch in range(args.epochs):
        for i in range(0, n_train, 1024):
            ix = tr[i:i + 1024]
            z_n = Z_n[ix]; z_t = Z_t[ix]
            with torch.no_grad():
                f_z = bundle.transition(z_t)        # forward
            recon = g(f_z)
            loss = F.l1_loss(recon, z_t) + (1 - F.cosine_similarity(recon, z_t).mean())
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def round_trip(h: int) -> float:
        z = Z_t[te]
        z_fwd = z
        for _ in range(h):
            z_fwd = bundle.transition(z_fwd)
        z_back = z_fwd
        for _ in range(h):
            z_back = g(z_back)
        return float(F.l1_loss(z_back, z).item())

    metrics = {"n_pairs": int(n), "round_trip_l1": {h: round_trip(h) for h in (1, 2, 4, 8)}}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
