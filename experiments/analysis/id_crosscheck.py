"""Cross-check TwoNN intrinsic dimension with the MLE estimator
(Levina & Bickel, 2004) using the cached Z_pooled.npy from the
persistent_homology runs. Reports both estimators side-by-side for the
five PH cells.

Outputs:
  paper/_figs/id_crosscheck.csv
  prints a markdown table
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def twonn_id(X: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    d, _ = tree.query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = mu[mu > 1.0]
    if len(mu) < 50:
        return float("nan")
    F = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    x = np.log(np.sort(mu))
    y = -np.log(1 - F)
    return float(np.sum(x * y) / np.sum(x * x))


def mle_id(X: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel MLE intrinsic dimension at neighbourhood size k.

    d_hat_k(x) = (1/(k-1)) sum_{j=1}^{k-1} log(T_k(x)/T_j(x))
    Final estimate: harmonic mean across points (their recommended form).
    """
    from scipy.spatial import cKDTree
    n = X.shape[0]
    tree = cKDTree(X)
    d, _ = tree.query(X, k=k + 1)  # self + k
    # drop self distance (column 0); columns 1..k are r_1..r_k
    R = d[:, 1:]
    R = np.maximum(R, 1e-12)
    log_ratios = np.log(R[:, -1:] / R[:, :-1])  # (n, k-1)
    inv_d = (1.0 / (k - 1)) * log_ratios.sum(axis=1)
    inv_d = inv_d[inv_d > 0]
    if inv_d.size == 0:
        return float("nan")
    # Inverse-mean form (MacKay & Ghahramani correction)
    return float(1.0 / inv_d.mean())


CELLS = [
    ("default (d128)",        "S1_dim/d128"),
    ("collapse (lambda=0)",   "S2_lambda_u/u0000"),
    ("default (lambda=0.1)",  "S2_lambda_u/u0100"),
    ("shuffled control",      "C1_shuffled_time/shuffled"),
    ("fine-tuned encoder",    "S4_encoder/ftuned"),
]

ROOT = Path("/mnt/vast/exp/brucexu/EchoJEPA/experiments_2026")
OUT = Path("/mnt/home/brucexu/nips/EchoJEPA/paper/_figs/id_crosscheck.csv")


def main() -> None:
    rows = [("cell", "n", "d_amb", "twonn", "mle_k5", "mle_k10", "mle_k20")]
    print(f"{'cell':<28} {'n':>5} {'d':>4} {'TwoNN':>7} {'MLE k=5':>9} {'MLE k=10':>10} {'MLE k=20':>10}")
    for name, rel in CELLS:
        zp = ROOT / rel / "analysis_ph" / "Z_pooled.npy"
        if not zp.exists():
            print(f"{name:<28} MISSING")
            continue
        Z = np.load(zp)
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        n, d = Zn.shape
        t = twonn_id(Zn)
        m5 = mle_id(Zn, k=5)
        m10 = mle_id(Zn, k=10)
        m20 = mle_id(Zn, k=20)
        rows.append((name, n, d, f"{t:.2f}", f"{m5:.2f}", f"{m10:.2f}", f"{m20:.2f}"))
        print(f"{name:<28} {n:>5d} {d:>4d} {t:>7.2f} {m5:>9.2f} {m10:>10.2f} {m20:>10.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
