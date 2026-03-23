import logging
import re
import sys

import torch

import src.models.predictor as vit_pred
import src.models.vision_transformer as video_vit
from src.models.sonostate import StateHead, Transition
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _strip_ddp_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {re.sub(r'^module\.(?:module\.)?', '', k): v for k, v in state_dict.items()}


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    state_dim=256,
    transition_hidden_dim=512,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
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
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    state_head = StateHead(embed_dim=embed_dim, state_dim=state_dim)
    transition = Transition(state_dim=state_dim, hidden_dim=transition_hidden_dim)

    encoder.to(device)
    predictor.to(device)
    state_head.to(device)
    transition.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor parameters: {count_parameters(predictor)}")
    logger.info(f"StateHead parameters: {count_parameters(state_head)}")
    logger.info(f"Transition parameters: {count_parameters(transition)}")

    return encoder, predictor, state_head, transition


def init_opt(
    encoder,
    predictor,
    state_head,
    transition,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    dtype=torch.float32,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    is_anneal=False,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters()
                       if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in predictor.named_parameters()
                       if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in encoder.named_parameters()
                       if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in predictor.named_parameters()
                       if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": state_head.parameters(),
        },
        {
            "params": transition.parameters(),
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler("cuda") if mixed_precision and dtype == torch.float16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    state_head,
    transition,
    opt,
    scaler,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]
    itr = checkpoint.get("itr", 0)

    pretrained_dict = _strip_ddp_prefix(checkpoint["encoder"])
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded encoder from epoch {epoch}: {msg}")

    pretrained_dict = _strip_ddp_prefix(checkpoint["predictor"])
    msg = predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded predictor from epoch {epoch}: {msg}")

    if target_encoder is not None and "target_encoder" in checkpoint:
        pretrained_dict = _strip_ddp_prefix(checkpoint["target_encoder"])
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded target encoder from epoch {epoch}: {msg}")

    if "state_head" in checkpoint:
        pretrained_dict = _strip_ddp_prefix(checkpoint["state_head"])
        msg = state_head.load_state_dict(pretrained_dict)
        logger.info(f"loaded state_head from epoch {epoch}: {msg}")

    if "transition" in checkpoint:
        pretrained_dict = _strip_ddp_prefix(checkpoint["transition"])
        msg = transition.load_state_dict(pretrained_dict)
        logger.info(f"loaded transition from epoch {epoch}: {msg}")

    opt.load_state_dict(checkpoint["opt"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        state_head,
        transition,
        opt,
        scaler,
        epoch,
        itr,
    )
