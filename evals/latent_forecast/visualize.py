"""
Visualize SonoState latent trajectories.

Usage:
    python -m evals.latent_forecast.visualize \
        --trajectories eval_output/trajectories.npz \
        --output eval_output/trajectory_plots
"""

import argparse
import os

import numpy as np


def plot_pca_trajectories(trajectories, labels, output_dir, n_components=2):
    """Project trajectories to 2D via PCA and plot."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    all_states = np.concatenate(trajectories, axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(all_states)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.cm.tab20
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        proj = pca.transform(traj)
        color = cmap(i % 20)
        ax.plot(proj[:, 0], proj[:, 1], '-o', color=color, markersize=3, linewidth=1, alpha=0.7)
        ax.plot(proj[0, 0], proj[0, 1], 's', color=color, markersize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("SonoState Latent Trajectories (PCA)")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "pca_trajectories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PCA plot to {path}")


def plot_umap_trajectories(trajectories, labels, output_dir):
    """Project trajectories to 2D via UMAP and plot."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed; skipping UMAP visualization. pip install umap-learn")
        return

    import matplotlib.pyplot as plt

    all_states = np.concatenate(trajectories, axis=0)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(all_states)

    # Split back per trajectory
    idx = 0
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.cm.tab20
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        n = len(traj)
        proj = embedding[idx:idx + n]
        idx += n
        color = cmap(i % 20)
        ax.plot(proj[:, 0], proj[:, 1], '-o', color=color, markersize=3, linewidth=1, alpha=0.7)
        ax.plot(proj[0, 0], proj[0, 1], 's', color=color, markersize=8)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("SonoState Latent Trajectories (UMAP)")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "umap_trajectories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP plot to {path}")


def plot_rollout_error(results_path, output_dir):
    """Plot forecast error vs horizon from results JSON."""
    import json
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        results = json.load(f)

    horizons, forecast_errs, persistence_errs = [], [], []
    h = 1
    while f"forecast_L1_h{h}" in results:
        horizons.append(h)
        forecast_errs.append(results[f"forecast_L1_h{h}"])
        persistence_errs.append(results[f"persistence_L1_h{h}"])
        h += 1

    if not horizons:
        print("No horizon data found in results.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(horizons, forecast_errs, '-o', label="Transition model", linewidth=2, markersize=6)
    ax.plot(horizons, persistence_errs, '--s', label="Persistence baseline", linewidth=2, markersize=6)
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("L1 Error")
    ax.set_title("SonoState Multi-step Rollout Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "rollout_error.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rollout error plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SonoState latent trajectories")
    parser.add_argument("--trajectories", type=str, required=True, help="Path to trajectories.npz")
    parser.add_argument("--results", type=str, default=None, help="Path to forecast_results.json")
    parser.add_argument("--output", type=str, default="eval_output/plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    data = np.load(args.trajectories, allow_pickle=True)
    trajectories = list(data["trajectories"])
    labels = list(data["labels"])
    print(f"Loaded {len(trajectories)} trajectories")

    plot_pca_trajectories(trajectories, labels, args.output)
    plot_umap_trajectories(trajectories, labels, args.output)

    if args.results:
        plot_rollout_error(args.results, args.output)


if __name__ == "__main__":
    main()
