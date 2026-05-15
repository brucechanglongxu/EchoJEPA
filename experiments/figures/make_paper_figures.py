"""Build all paper-ready figures and tables from analysis outputs.

Globs `metrics.json` files under EXP_ROOT and produces:

  paper/_figs/forecast_curves.pdf
  paper/_figs/phase_recovery.pdf
  paper/_figs/dim_sweep.pdf
  paper/_figs/uniformity_sweep.pdf
  paper/_figs/init_ablation.pdf
  paper/_figs/shuffled_control.pdf
  paper/_figs/RESULTS.md          (paper-ready markdown tables)

Run after `condor_submit_dag experiments/condor/sweeps.dag` finishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_all(root: Path) -> pd.DataFrame:
    rows = []
    for mj in root.glob("*/*/analysis_*/metrics.json"):
        try:
            data = json.loads(mj.read_text())
        except Exception:
            continue
        sweep = mj.parts[-4]
        cell = mj.parts[-3]
        analysis = mj.parts[-2].replace("analysis_", "")
        rows.append({
            "sweep": sweep, "cell": cell, "analysis": analysis,
            "metrics": data, "path": str(mj),
        })
    return pd.DataFrame(rows)


def _flatten(df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    sub = df[df.analysis == analysis].copy()
    flat = pd.json_normalize(sub["metrics"])
    flat["sweep"] = sub["sweep"].values
    flat["cell"] = sub["cell"].values
    return flat


def plot_dim_sweep(df: pd.DataFrame, out: Path):
    forc = _flatten(df, "forecast_curves")
    phase = _flatten(df, "phase_decoding")
    geom = _flatten(df, "geometry")
    sub = forc[forc.sweep == "S1_dim"].copy() if not forc.empty else forc
    if sub.empty or "horizons.1.forecast_l1" not in sub.columns:
        return
    sub["d"] = sub.cell.str.replace("d", "").astype(int)
    sub = sub.sort_values("d")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(sub.d, sub["horizons.1.forecast_l1"], "s-", label="SonoState")
    axes[0].plot(sub.d, sub["horizons.1.persistence_l1"], "o--", label="persist.")
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("latent dim d")
    axes[0].set_ylabel("h=1 L1 error"); axes[0].legend()

    p = phase[phase.sweep == "S1_dim"].copy() if not phase.empty else phase
    if not p.empty and "ed_vs_es_classification_acc" in p.columns:
        p["d"] = p.cell.str.replace("d", "").astype(int)
        p = p.sort_values("d")
        axes[1].plot(p.d, p["ed_vs_es_classification_acc"], "o-")
        axes[1].axhline(0.5, color="grey", ls="--", label="chance")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("latent dim d")
        axes[1].set_ylabel("ED-vs-ES decode accuracy"); axes[1].legend()

    g = geom[geom.sweep == "S1_dim"].copy()
    if not g.empty and "intrinsic_dim_twonn" in g.columns:
        g["d"] = g.cell.str.replace("d", "").astype(int)
        g = g.sort_values("d")
        axes[2].plot(g.d, g["intrinsic_dim_twonn"], "o-", label="TwoNN ID")
        axes[2].plot(g.d, g.d, "k:", label="d (ambient)")
        axes[2].set_xscale("log", base=2); axes[2].set_yscale("log", base=2)
        axes[2].set_xlabel("latent dim d"); axes[2].set_ylabel("intrinsic dim")
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(out / "dim_sweep.pdf"); plt.close(fig)


def plot_uniformity(df: pd.DataFrame, out: Path):
    forc = _flatten(df, "forecast_curves")
    phase = _flatten(df, "phase_decoding")
    sub = forc[forc.sweep == "S2_lambda_u"].copy() if not forc.empty else forc
    if sub.empty or "horizons.1.forecast_l1" not in sub.columns:
        return
    sub["lu"] = sub.cell.str.replace("u", "").astype(int) / 100.0
    sub = sub.sort_values("lu")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(sub.lu, sub["horizons.1.forecast_l1"], "s-", label="SonoState")
    axes[0].plot(sub.lu, sub["horizons.1.persistence_l1"], "o--", label="persist.")
    axes[0].set_xlabel(r"$\lambda_u$"); axes[0].set_ylabel("h=1 L1"); axes[0].legend()
    p = phase[phase.sweep == "S2_lambda_u"].copy() if not phase.empty else phase
    if not p.empty and "ed_vs_es_classification_acc" in p.columns:
        p["lu"] = p.cell.str.replace("u", "").astype(int) / 100.0
        p = p.sort_values("lu")
        axes[1].plot(p.lu, p["ed_vs_es_classification_acc"], "o-")
        axes[1].axhline(0.5, color="grey", ls="--")
        axes[1].set_xlabel(r"$\lambda_u$")
        axes[1].set_ylabel("ED vs ES acc")
    fig.tight_layout(); fig.savefig(out / "uniformity_sweep.pdf"); plt.close(fig)


def plot_init_ablation(df: pd.DataFrame, out: Path):
    forc = _flatten(df, "forecast_curves")
    sub = forc[forc.sweep == "S5_init"].copy() if not forc.empty else forc
    if sub.empty or "horizons.1.forecast_l1" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    cells = sub.cell.tolist()
    h1 = sub["horizons.1.forecast_l1"].tolist()
    persist = sub["horizons.1.persistence_l1"].tolist()
    x = np.arange(len(cells)); w = 0.35
    ax.bar(x - w/2, h1, w, label="SonoState")
    ax.bar(x + w/2, persist, w, label="persist.")
    ax.set_xticks(x); ax.set_xticklabels(cells)
    ax.set_ylabel("h=1 L1 error"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "init_ablation.pdf"); plt.close(fig)


def plot_shuffled_control(df: pd.DataFrame, out: Path):
    forc = _flatten(df, "forecast_curves")
    sub = forc[forc.sweep == "C1_shuffled_time"].copy() if not forc.empty else forc
    if sub.empty or "horizons.1.forecast_l1" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    h1 = sub.set_index("cell")["horizons.1.forecast_l1"]
    persist = sub.set_index("cell")["horizons.1.persistence_l1"]
    h1.plot(kind="bar", ax=ax, label="forecast", position=0, width=0.4, color="tab:blue")
    persist.plot(kind="bar", ax=ax, label="persistence", position=1, width=0.4, color="tab:grey")
    ax.set_ylabel("h=1 L1"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "shuffled_control.pdf"); plt.close(fig)


def write_results_md(df: pd.DataFrame, out: Path):
    lines = ["# Auto-generated results", ""]
    for analysis in sorted(df.analysis.unique()):
        lines.append(f"## {analysis}")
        flat = _flatten(df, analysis)
        cols = [c for c in flat.columns if c not in {"sweep", "cell"}]
        lines.append(flat[["sweep", "cell"] + cols].to_markdown(index=False))
        lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="EXP_ROOT containing {sweep}/{cell}/analysis_*/metrics.json")
    p.add_argument("--out", required=True, help="Where to write figures and RESULTS.md")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = _load_all(Path(args.root))
    if df.empty:
        print("no metrics.json files found"); return
    print(f"loaded {len(df)} metrics.json files across analyses: "
          f"{sorted(df.analysis.unique())}")
    for fn in (plot_dim_sweep, plot_uniformity, plot_init_ablation, plot_shuffled_control):
        try:
            fn(df, out)
        except Exception as e:
            print(f"[skip] {fn.__name__}: {e}")
    write_results_md(df, out)
    # Also dump a wide CSV per analysis for easy paper consumption.
    for a in sorted(df.analysis.unique()):
        flat = _flatten(df, a)
        flat.to_csv(out / f"{a}.csv", index=False)
    print(f"wrote figures + RESULTS.md to {out}")


if __name__ == "__main__":
    main()
