"""
SonoState latent forecast evaluation.

Evaluates:
1. Single-step forecast accuracy: ||f(z_t) - z_{t+1}||
2. Multi-step rollout error vs horizon
3. Persistence baseline comparison: ||z_t - z_{t+1}||
4. Latent trajectory structure (PCA/UMAP of z_t over full videos)
"""

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from src.datasets.data_manager import init_data
from src.models.sonostate import StateHead, Transition
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger

import src.models.vision_transformer as video_vit
from src.utils.wrappers import MultiSeqWrapper

logging.basicConfig()
logger = get_logger(__name__, force=True)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def _strip_ddp_prefix(state_dict):
    import re
    return {re.sub(r'^module\.(?:module\.)?', '', k): v for k, v in state_dict.items()}


def load_sonostate_models(checkpoint_path, device, model_cfg, sonostate_cfg):
    """Load encoder (target), state_head, and transition from a SonoState checkpoint."""
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))

    model_name = model_cfg["model_name"]
    crop_size = model_cfg.get("crop_size", 224)
    patch_size = model_cfg.get("patch_size", 16)
    num_frames = model_cfg.get("num_frames", 16)
    tubelet_size = model_cfg.get("tubelet_size", 2)
    uniform_power = model_cfg.get("uniform_power", False)
    use_sdpa = model_cfg.get("use_sdpa", True)
    use_silu = model_cfg.get("use_silu", False)
    wide_silu = model_cfg.get("wide_silu", True)
    use_rope = model_cfg.get("use_rope", True)
    use_activation_checkpointing = model_cfg.get("use_activation_checkpointing", False)

    state_dim = sonostate_cfg.get("state_dim", 256)
    transition_hidden_dim = sonostate_cfg.get("transition_hidden_dim", 512)

    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    embed_dim = encoder.embed_dim
    encoder = MultiSeqWrapper(encoder)

    state_head = StateHead(embed_dim=embed_dim, state_dim=state_dim)
    transition = Transition(state_dim=state_dim, hidden_dim=transition_hidden_dim)

    # Prefer target_encoder weights (EMA-smoothed)
    encoder_key = "target_encoder" if "target_encoder" in checkpoint else "encoder"
    pretrained = _strip_ddp_prefix(checkpoint[encoder_key])
    msg = encoder.load_state_dict(pretrained, strict=False)
    logger.info(f"Loaded encoder ({encoder_key}): {msg}")

    if "state_head" in checkpoint:
        pretrained = _strip_ddp_prefix(checkpoint["state_head"])
        state_head.load_state_dict(pretrained)
        logger.info("Loaded state_head")

    if "transition" in checkpoint:
        pretrained = _strip_ddp_prefix(checkpoint["transition"])
        transition.load_state_dict(pretrained)
        logger.info("Loaded transition")

    encoder.to(device).eval()
    state_head.to(device).eval()
    transition.to(device).eval()

    del checkpoint
    return encoder, state_head, transition


def make_eval_transforms(crop_size=224):
    """Simple eval transforms: resize + center crop + normalize."""
    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms

    short_side_size = int(crop_size * 256 / 224)
    transform = video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform


@torch.no_grad()
def evaluate_forecast(
    encoder,
    state_head,
    transition,
    data_loader,
    device,
    max_horizon=10,
    dtype=torch.bfloat16,
):
    """Compute single- and multi-step forecast error + persistence baseline."""
    forecast_errors = {h: AverageMeter() for h in range(1, max_horizon + 1)}
    persistence_errors = {h: AverageMeter() for h in range(1, max_horizon + 1)}
    n_videos = 0

    for batch in data_loader:
        clips_list, labels, clip_indices = batch
        if not isinstance(clips_list, list):
            clips_list = [clips_list]

        # Each clip in clips_list is [B, C, T, H, W]; they come from a multi-clip dataset
        # We need at least max_horizon+1 clips for rollout eval
        num_clips = len(clips_list)
        if num_clips < 2:
            continue

        with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            states = []
            for clip in clips_list:
                clip = clip.to(device, non_blocking=True)
                tokens = encoder([clip])
                z = state_head(tokens[0])
                states.append(z)

            # Single- and multi-step evaluation
            max_h = min(max_horizon, num_clips - 1)
            for t in range(num_clips - 1):
                z_t = states[t]
                z_rolled = z_t
                for h in range(1, max_h + 1):
                    if t + h >= num_clips:
                        break
                    z_rolled = transition(z_rolled)
                    z_true = states[t + h]
                    err = F.l1_loss(z_rolled, z_true).item()
                    persistence_err = F.l1_loss(z_t, z_true).item()
                    forecast_errors[h].update(err)
                    persistence_errors[h].update(persistence_err)

            n_videos += clips_list[0].size(0)

    results = {}
    for h in range(1, max_horizon + 1):
        if forecast_errors[h].count > 0:
            results[f"forecast_L1_h{h}"] = forecast_errors[h].avg
            results[f"persistence_L1_h{h}"] = persistence_errors[h].avg
            results[f"improvement_h{h}"] = persistence_errors[h].avg - forecast_errors[h].avg

    results["n_videos"] = n_videos
    return results


@torch.no_grad()
def extract_trajectories(
    encoder,
    state_head,
    data_loader,
    device,
    max_videos=50,
    dtype=torch.bfloat16,
):
    """Extract latent state trajectories over full videos for visualization."""
    all_trajectories = []
    all_labels = []
    n_collected = 0

    for batch in data_loader:
        if n_collected >= max_videos:
            break

        clips_list, labels, clip_indices = batch
        if not isinstance(clips_list, list):
            clips_list = [clips_list]

        with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            batch_states = []
            for clip in clips_list:
                clip = clip.to(device, non_blocking=True)
                tokens = encoder([clip])
                z = state_head(tokens[0])
                batch_states.append(z.cpu().float())

            # states shape: (num_clips, B, d_state)
            trajectory = torch.stack(batch_states, dim=1)  # (B, num_clips, d_state)

            for b in range(trajectory.size(0)):
                if n_collected >= max_videos:
                    break
                all_trajectories.append(trajectory[b].numpy())
                all_labels.append(labels[b].item() if torch.is_tensor(labels) else labels[b])
                n_collected += 1

    return all_trajectories, all_labels


def main(args_eval, resume_preempt=False):
    """Entry point for SonoState latent forecast evaluation."""
    logger.info("Starting SonoState latent forecast evaluation")

    checkpoint_path = args_eval["checkpoint"]
    output_dir = args_eval.get("output_dir", "eval_output")
    os.makedirs(output_dir, exist_ok=True)

    model_cfg = args_eval.get("model", {})
    sonostate_cfg = args_eval.get("sonostate", {})
    data_cfg = args_eval.get("data", {})
    max_horizon = args_eval.get("max_horizon", 10)

    crop_size = model_cfg.get("crop_size", 224)
    dtype_str = args_eval.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    world_size, rank = init_distributed()

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    encoder, state_head, transition = load_sonostate_models(
        checkpoint_path, device, model_cfg, sonostate_cfg,
    )

    transform = make_eval_transforms(crop_size=crop_size)

    num_clips = data_cfg.get("num_clips", max_horizon + 1)
    (eval_loader, eval_sampler) = init_data(
        data=data_cfg.get("dataset_type", "videodataset"),
        root_path=data_cfg.get("datasets", []),
        batch_size=data_cfg.get("batch_size", 8),
        training=False,
        dataset_fpcs=data_cfg.get("dataset_fpcs", [16]),
        fps=data_cfg.get("fps", 8),
        num_clips=num_clips,
        transform=transform,
        rank=rank,
        world_size=world_size,
        num_workers=data_cfg.get("num_workers", 4),
        pin_mem=True,
        persistent_workers=False,
        drop_last=False,
    )

    # 1. Forecast accuracy
    logger.info("Evaluating forecast accuracy...")
    results = evaluate_forecast(
        encoder, state_head, transition, eval_loader, device,
        max_horizon=min(max_horizon, num_clips - 1),
        dtype=dtype,
    )

    logger.info("=== Forecast Results ===")
    for k, v in sorted(results.items()):
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    results_path = os.path.join(output_dir, "forecast_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # 2. Trajectory extraction (rank 0 only)
    if rank == 0:
        logger.info("Extracting latent trajectories...")
        trajectories, labels = extract_trajectories(
            encoder, state_head, eval_loader, device,
            max_videos=50,
            dtype=dtype,
        )
        traj_path = os.path.join(output_dir, "trajectories.npz")
        np.savez(traj_path, trajectories=np.array(trajectories, dtype=object), labels=np.array(labels))
        logger.info(f"Trajectories saved to {traj_path} ({len(trajectories)} videos)")

    logger.info("Evaluation complete.")
    return results
