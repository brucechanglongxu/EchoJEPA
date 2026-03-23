"""
SonoState Proof-of-Concept Evaluation

Generates presentation-ready figures:
  1. Training loss curves (from CSV log)
  2. Latent trajectory PCA (cardiac cycle loops)
  3. Single-step forecast error vs persistence baseline
  4. Multi-step rollout error vs horizon

Usage:
    PYTHONPATH=/scratch/bxu/project/EchoJEPA python3 scripts/sonostate_poc_eval.py \
        --checkpoint /scratch/bxu/project/EchoJEPA/checkpoints/sonostate_poc/latest.pt \
        --data_csv /scratch/bxu/project/echonet/pretrain_annotations.csv \
        --output /scratch/bxu/project/EchoJEPA/checkpoints/sonostate_poc/figures
"""

import argparse
import csv
import os
import re
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

import src.models.vision_transformer as video_vit
from src.models.sonostate import StateHead, Transition
from src.utils.wrappers import MultiSeqWrapper
from src.utils.checkpoint_loader import robust_checkpoint_loader

warnings.filterwarnings("ignore")


def strip_ddp(state_dict):
    return {re.sub(r'^module\.(?:module\.)?', '', k): v for k, v in state_dict.items()}


def load_models(ckpt_path, device):
    ckpt = robust_checkpoint_loader(ckpt_path, map_location="cpu")
    encoder = video_vit.vit_large(
        img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
        uniform_power=True, use_sdpa=True, use_rope=True,
        use_activation_checkpointing=False,
    )
    embed_dim = encoder.embed_dim
    encoder = MultiSeqWrapper(encoder)
    state_head = StateHead(embed_dim=embed_dim, state_dim=256)
    transition = Transition(state_dim=256, hidden_dim=512)

    key = "target_encoder" if "target_encoder" in ckpt else "encoder"
    encoder.load_state_dict(strip_ddp(ckpt[key]), strict=False)
    state_head.load_state_dict(strip_ddp(ckpt["state_head"]), strict=False)
    transition.load_state_dict(strip_ddp(ckpt["transition"]), strict=False)

    encoder.to(device).eval()
    state_head.to(device).eval()
    transition.to(device).eval()
    del ckpt
    return encoder, state_head, transition


def load_video_clips(video_path, num_clips=10, fpc=16, fps_target=8):
    """Load a video and split into overlapping sliding-window clips."""
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    video_fps = round(vr.get_avg_fps())
    step = max(1, video_fps // fps_target)
    total_frames = len(vr)
    clip_len = fpc * step
    # Use 50% overlap so short echo videos yield enough clips
    stride = max(1, clip_len // 2)

    clips = []
    for start in range(0, total_frames - clip_len + 1, stride):
        indices = np.linspace(start, start + clip_len - 1, num=fpc).astype(np.int64)
        indices = np.clip(indices, 0, total_frames - 1)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
        clips.append(frames)
        if len(clips) >= num_clips:
            break
    return clips


def preprocess_clip(frames, crop_size=224):
    """Convert raw frames (T, H, W, 3) to tensor (1, C, T, H, W)."""
    t = torch.tensor(frames, dtype=torch.float32)
    t = t.permute(3, 0, 1, 2)  # C, T, H, W
    # Resize
    C, T, H, W = t.shape
    t = t.reshape(C * T, 1, H, W).expand(-1, 1, -1, -1)
    t = F.interpolate(
        t.reshape(C * T, 1, H, W), size=(crop_size, crop_size),
        mode='bilinear', align_corners=False,
    ).reshape(C, T, crop_size, crop_size)
    # Normalize (ImageNet stats, 0-255 scale)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1) * 255.0
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1) * 255.0
    t = (t - mean) / std
    return t.unsqueeze(0)  # (1, C, T, H, W)


@torch.no_grad()
def extract_states(encoder, state_head, clips, device):
    """Extract latent states from a list of clips."""
    states = []
    for clip_frames in clips:
        x = preprocess_clip(clip_frames).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder([x])
            z = state_head(tokens[0])
        states.append(z.cpu().float().squeeze(0))
    return torch.stack(states)  # (num_clips, d_state)


def plot_loss_curves(log_dir, output_dir):
    """Parse CSV log and plot training loss curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log_file = os.path.join(log_dir, "log_r0.csv")
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    epochs, itrs, losses, jepa_losses, forecast_losses = [], [], [], [], []
    with open(log_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5:
                try:
                    epochs.append(int(row[0]))
                    itrs.append(int(row[1]))
                    losses.append(float(row[2]))
                    jepa_losses.append(float(row[3]))
                    forecast_losses.append(float(row[4]))
                except ValueError:
                    continue

    steps = list(range(len(losses)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(steps, losses, color='#2196F3', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, jepa_losses, color='#4CAF50', linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('JEPA Loss', fontsize=12)
    axes[1].set_title('JEPA Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, forecast_losses, color='#FF5722', linewidth=1.5, alpha=0.8)
    axes[2].set_xlabel('Training Step', fontsize=12)
    axes[2].set_ylabel('Forecast Loss (L1)', fontsize=12)
    axes[2].set_title('State Transition Forecast Loss', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_training_losses.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_trajectories_pca(all_states, video_names, output_dir):
    """PCA projection of latent trajectories — the key visual."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    all_points = np.concatenate([s.numpy() for s in all_states], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_points)

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    cmap = plt.cm.Set2
    for i, (states, name) in enumerate(zip(all_states, video_names)):
        proj = pca.transform(states.numpy())
        color = cmap(i % 8)
        ax.plot(proj[:, 0], proj[:, 1], '-', color=color, linewidth=2, alpha=0.7)
        ax.scatter(proj[:, 0], proj[:, 1], color=color, s=30, zorder=5, alpha=0.8)
        ax.scatter(proj[0, 0], proj[0, 1], color=color, s=120, marker='s',
                   edgecolors='black', linewidth=1.5, zorder=6, label=f'Video {i+1}')

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=13)
    ax.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=13)
    ax.set_title('SonoState Latent Trajectories\n(Each path = one echocardiogram video)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "02_latent_trajectories_pca.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_forecast_vs_persistence(all_states, transition, device, output_dir):
    """Forecast error vs persistence baseline at multiple horizons."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    max_horizon = 6
    forecast_errors = {h: [] for h in range(1, max_horizon + 1)}
    persistence_errors = {h: [] for h in range(1, max_horizon + 1)}

    for states in all_states:
        n = states.shape[0]
        for t in range(n - 1):
            z_t = states[t:t+1].to(device)
            z_rolled = z_t.clone()
            for h in range(1, max_horizon + 1):
                if t + h >= n:
                    break
                z_true = states[t+h:t+h+1].to(device)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_rolled = transition(z_rolled)
                f_err = F.l1_loss(z_rolled.float(), z_true.float()).item()
                p_err = F.l1_loss(z_t.float(), z_true.float()).item()
                forecast_errors[h].append(f_err)
                persistence_errors[h].append(p_err)

    horizons = []
    f_means, p_means = [], []
    for h in range(1, max_horizon + 1):
        if forecast_errors[h]:
            horizons.append(h)
            f_means.append(np.mean(forecast_errors[h]))
            p_means.append(np.mean(persistence_errors[h]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(horizons, f_means, '-o', color='#2196F3', linewidth=2.5, markersize=8,
            label='SonoState Transition', zorder=5)
    ax.plot(horizons, p_means, '--s', color='#9E9E9E', linewidth=2.5, markersize=8,
            label='Persistence Baseline (z\u0302 = z_t)', zorder=4)
    ax.fill_between(horizons, f_means, p_means, alpha=0.15, color='#2196F3')
    ax.set_xlabel('Forecast Horizon (clips)', fontsize=13)
    ax.set_ylabel('L1 Error', fontsize=13)
    ax.set_title('Multi-step Rollout: SonoState vs Persistence',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)

    if f_means and p_means:
        improvement = ((p_means[0] - f_means[0]) / p_means[0]) * 100
        ax.annotate(f'{improvement:.0f}% better at h=1',
                    xy=(1, f_means[0]), xytext=(2, f_means[0] - 0.002),
                    fontsize=11, color='#2196F3', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2196F3'))

    path = os.path.join(output_dir, "03_forecast_vs_persistence.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Print table
    print("\n" + "="*55)
    print("  Horizon | Transition | Persistence | Improvement")
    print("-"*55)
    for h, f, p in zip(horizons, f_means, p_means):
        imp = ((p - f) / p) * 100
        print(f"     {h}    |   {f:.4f}   |   {p:.4f}    |   {imp:+.1f}%")
    print("="*55)


def plot_single_trajectory_detail(states, transition, device, output_dir):
    """Single video: true vs predicted trajectory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n = states.shape[0]
    pca = PCA(n_components=2)
    true_proj = pca.fit_transform(states.numpy())

    z0 = states[0:1].to(device)
    rollout = [z0.cpu().float().squeeze(0)]
    z = z0
    with torch.no_grad():
        for _ in range(n - 1):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = transition(z)
            rollout.append(z.cpu().float().squeeze(0))
    rollout = torch.stack(rollout).numpy()
    pred_proj = pca.transform(rollout)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(true_proj[:, 0], true_proj[:, 1], '-o', color='#4CAF50', linewidth=2,
            markersize=6, label='True trajectory', zorder=5)
    ax.plot(pred_proj[:, 0], pred_proj[:, 1], '--^', color='#FF5722', linewidth=2,
            markersize=6, label='Rollout (from z₀ only)', zorder=4)
    ax.scatter(true_proj[0, 0], true_proj[0, 1], color='black', s=150, marker='*',
               zorder=6, label='Start (z₀)')

    for i in range(n):
        ax.annotate(f't={i}', (true_proj[i, 0], true_proj[i, 1]),
                    fontsize=7, alpha=0.6, ha='center', va='bottom')

    ax.set_xlabel('PC1', fontsize=13)
    ax.set_ylabel('PC2', fontsize=13)
    ax.set_title('True vs Predicted Latent Trajectory (Single Video)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "04_true_vs_predicted_trajectory.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--output", default="figures")
    parser.add_argument("--num_videos", type=int, default=8)
    parser.add_argument("--num_clips", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 1. Loss curves from training log
    log_dir = os.path.dirname(args.checkpoint)
    print("\n=== Plotting training loss curves ===")
    plot_loss_curves(log_dir, args.output)

    # 2. Load models
    print("\n=== Loading SonoState models ===")
    encoder, state_head, transition = load_models(args.checkpoint, device)

    # 3. Load videos and extract states
    print(f"\n=== Extracting latent states from {args.num_videos} videos ===")
    with open(args.data_csv) as f:
        lines = f.readlines()
    np.random.seed(42)
    selected = np.random.choice(len(lines), size=min(args.num_videos, len(lines)), replace=False)

    all_states = []
    video_names = []
    for idx in selected:
        path = lines[idx].strip().split()[0]
        name = os.path.basename(path)
        try:
            clips = load_video_clips(path, num_clips=args.num_clips)
            if len(clips) < 2:
                continue
            states = extract_states(encoder, state_head, clips, device)
            all_states.append(states)
            video_names.append(name)
            print(f"  {name}: {len(clips)} clips → states {states.shape}")
        except Exception as e:
            print(f"  Skipped {name}: {e}")

    if not all_states:
        print("No videos processed successfully!")
        return

    # 4. Plot latent trajectories (PCA)
    print("\n=== Plotting latent trajectories ===")
    plot_trajectories_pca(all_states, video_names, args.output)

    # 5. Forecast vs persistence
    print("\n=== Computing forecast vs persistence ===")
    plot_forecast_vs_persistence(all_states, transition, device, args.output)

    # 6. Single trajectory: true vs rollout
    print("\n=== Plotting true vs predicted trajectory ===")
    longest = max(range(len(all_states)), key=lambda i: all_states[i].shape[0])
    plot_single_trajectory_detail(all_states[longest], transition, device, args.output)

    print(f"\n{'='*55}")
    print(f"All figures saved to: {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
