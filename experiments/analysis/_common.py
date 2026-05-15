"""Common loader / extractor utilities for SonoState analysis scripts.

All analysis scripts in this directory share the same plumbing:
  1. Load the encoder + state head + transition from a checkpoint.
  2. Iterate a CSV-driven video dataset.
  3. Extract per-clip latent states z_t.

We avoid any dependency on the training app to keep eval lightweight.
"""

from __future__ import annotations

import logging
import os
import re


def resolve_video_path(filename: str, video_root: str) -> str:
    """Resolve an EchoNet-style ``FileName`` into an absolute video path.

    EchoNet ``FileList.csv``/``VolumeTracings.csv`` use bare basenames
    (no ``.avi`` extension). Tolerate both bare and absolute inputs.
    """
    if os.path.isabs(filename):
        return filename
    p = os.path.join(video_root, filename)
    if os.path.isfile(p):
        return p
    # FileName like "0X100009310A3BD7FC" -> add .avi
    if not p.lower().endswith(".avi"):
        p_avi = p + ".avi"
        if os.path.isfile(p_avi):
            return p_avi
    return p
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

import src.models.vision_transformer as video_vit
from src.models.sonostate import StateHead, Transition
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.wrappers import MultiSeqWrapper

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

log = logging.getLogger("sonostate.analysis")
log.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _strip(state_dict: dict) -> dict:
    return {re.sub(r"^module\.(?:module\.)?", "", k): v for k, v in state_dict.items()}


@dataclass
class SonoStateBundle:
    encoder: torch.nn.Module
    state_head: StateHead
    transition: Transition
    embed_dim: int
    state_dim: int


def load_sonostate(
    checkpoint_path: str,
    device: torch.device,
    *,
    model_name: str = "vit_large",
    crop_size: int = 224,
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    use_rope: bool = True,
    state_dim: int = 256,
    transition_hidden_dim: int = 512,
    use_target_encoder: bool = True,
    transition_zero_init: bool = True,
    transition_residual: bool = True,
    transition_init_scale: float = -4.6,
) -> SonoStateBundle:
    ckpt = robust_checkpoint_loader(checkpoint_path, map_location="cpu")
    # Auto-detect state_dim and transition_hidden_dim from the saved
    # head/transition weights so a single analysis script works across
    # the entire S1_dim sweep.
    if "state_head" in ckpt:
        sh_sd = _strip(ckpt["state_head"])
        if "proj.weight" in sh_sd:
            state_dim = int(sh_sd["proj.weight"].shape[0])
    if "transition" in ckpt:
        tr_sd = _strip(ckpt["transition"])
        if "net.0.weight" in tr_sd:
            transition_hidden_dim = int(tr_sd["net.0.weight"].shape[0])
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=True,
        use_sdpa=True,
        use_rope=use_rope,
        use_activation_checkpointing=False,
    )
    embed_dim = encoder.embed_dim
    encoder = MultiSeqWrapper(encoder)
    state_head = StateHead(embed_dim=embed_dim, state_dim=state_dim)
    transition = Transition(
        state_dim=state_dim,
        hidden_dim=transition_hidden_dim,
        zero_init=transition_zero_init,
        residual=transition_residual,
        init_scale=transition_init_scale,
    )

    enc_key = "target_encoder" if (use_target_encoder and "target_encoder" in ckpt) else "encoder"
    encoder.load_state_dict(_strip(ckpt[enc_key]), strict=False)
    if "state_head" in ckpt:
        state_head.load_state_dict(_strip(ckpt["state_head"]), strict=False)
    if "transition" in ckpt:
        transition.load_state_dict(_strip(ckpt["transition"]), strict=False)

    encoder.to(device).eval()
    state_head.to(device).eval()
    transition.to(device).eval()
    del ckpt
    return SonoStateBundle(encoder, state_head, transition, embed_dim, state_dim)


# -----------------------------------------------------------------------------
# Standard eval transforms
# -----------------------------------------------------------------------------

def make_eval_transform(crop_size: int = 224):
    short_side = int(crop_size * 256 / 224)
    return video_transforms.Compose([
        video_transforms.Resize(short_side, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


# -----------------------------------------------------------------------------
# Per-video clip extraction (sliding window)
# -----------------------------------------------------------------------------

def sliding_clip_indices(
    n_frames: int,
    fps_video: int,
    fps_target: int = 8,
    fpc: int = 16,
    stride_frames: int | None = None,
) -> list[np.ndarray]:
    """Return a list of frame-index arrays, one per clip."""
    step = max(1, fps_video // fps_target)
    clip_len = fpc * step
    if stride_frames is None:
        stride_frames = max(1, clip_len // 2)
    out = []
    for start in range(0, max(1, n_frames - clip_len + 1), stride_frames):
        idx = np.linspace(start, start + clip_len - 1, num=fpc).astype(np.int64)
        idx = np.clip(idx, 0, n_frames - 1)
        out.append(idx)
    return out


@torch.no_grad()
def extract_video_states(
    bundle: SonoStateBundle,
    video_path: str,
    device: torch.device,
    crop_size: int = 224,
    fpc: int = 16,
    fps_target: int = 8,
    stride_frames: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[np.ndarray, list[int]]:
    """Read a video file and return (states[N,d], clip_center_frames)."""
    from decord import VideoReader, cpu  # local import
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    fps_video = max(1, round(vr.get_avg_fps()))
    n_frames = len(vr)
    idx_list = sliding_clip_indices(n_frames, fps_video, fps_target, fpc, stride_frames)
    if not idx_list:
        return np.empty((0, bundle.state_dim), dtype=np.float32), []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1, 1) * 255.0
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1, 1) * 255.0

    states: list[np.ndarray] = []
    centers: list[int] = []
    for indices in idx_list:
        frames = vr.get_batch(indices).asnumpy()             # T,H,W,3
        t = torch.from_numpy(frames).to(device, non_blocking=True).float()
        t = t.permute(3, 0, 1, 2)                            # C,T,H,W
        C, T, H, W = t.shape
        t = F.interpolate(
            t.reshape(C * T, 1, H, W), size=(crop_size, crop_size),
            mode="bilinear", align_corners=False,
        ).reshape(C, T, crop_size, crop_size)
        t = (t - mean) / std
        x = t.unsqueeze(0)                                    # 1,C,T,H,W
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            tokens = bundle.encoder([x])
            z = bundle.state_head(tokens[0])
        states.append(z.detach().float().cpu().numpy().squeeze(0))
        centers.append(int(indices[len(indices) // 2]))
    return np.stack(states, axis=0), centers


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s][%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
