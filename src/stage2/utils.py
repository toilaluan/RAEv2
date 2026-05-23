"""Stage 2 shared utilities.

Contains config validation + helpers shared between stage2/engine.py
"""

from __future__ import annotations

import dataclasses
from copy import deepcopy

import torch
from torch.cuda.amp import autocast

from configs.stage2 import Stage2Config
from stage2.models.embedders import TextEncoder
from utils.dist_utils import main_process_first


def validate_stage2_config(config: Stage2Config) -> None:
    """Validate a Stage2Config for consistency."""
    if not config.stage_1.target:
        raise ValueError("Config must provide stage_1.target (RAE model).")
    if not config.stage_2.target:
        raise ValueError("Config must provide stage_2.target (DiT model).")

    params = config.stage_2.params
    latent_layout = params.get("latent_layout", "2d")
    if latent_layout not in ("1d", "2d"):
        raise ValueError("stage_2.params.latent_layout must be '1d' or '2d'.")

    latent_size = list(config.misc.latent_size)
    input_size = params.get("input_size")
    in_channels = params.get("in_channels")
    if input_size is None:
        raise ValueError("stage_2.params.input_size is required.")
    if in_channels is None:
        raise ValueError("stage_2.params.in_channels is required.")

    patch_size = params.get("patch_size", [1, 1])
    if not isinstance(patch_size, (list, tuple)):
        patch_size = [patch_size, patch_size]

    if latent_layout == "1d":
        if len(latent_size) != 2:
            raise ValueError("latent_layout='1d' requires misc.latent_size=[num_tokens, channels].")
        num_tokens, channels = latent_size
        if input_size != num_tokens or in_channels != channels:
            raise ValueError(
                "For latent_layout='1d', stage_2.params.input_size/in_channels must match "
                f"misc.latent_size. Got input_size={input_size}, in_channels={in_channels}, "
                f"latent_size={latent_size}."
            )
        if any(p != 1 for p in patch_size):
            raise ValueError("latent_layout='1d' requires stage_2.params.patch_size to be [1, 1].")
        if config.conditioning.type == "nwm":
            raise ValueError("conditioning.type='nwm' is not supported with latent_layout='1d'.")
    else:
        if len(latent_size) != 3:
            raise ValueError("latent_layout='2d' requires misc.latent_size=[channels, height, width].")
        channels, height, width = latent_size
        if height != width:
            raise ValueError("DDT latent_layout='2d' requires square latents.")
        if input_size != height or input_size != width or in_channels != channels:
            raise ValueError(
                "For latent_layout='2d', stage_2.params.input_size/in_channels must match "
                f"misc.latent_size. Got input_size={input_size}, in_channels={in_channels}, "
                f"latent_size={latent_size}."
            )
        if any(input_size % p != 0 for p in patch_size):
            raise ValueError("stage_2.params.patch_size must divide stage_2.params.input_size.")

    # REPA validation
    repa = config.repa
    if repa.use_repa:
        if not repa.target_encoder:
            raise ValueError("repa.target_encoder is required when use_repa=True.")

    # Gradient accumulation
    if config.training.grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be >= 1.")

    # Conditioning
    cond = config.conditioning
    if cond.type == "text" and cond.text_encoder is None:
        raise ValueError("conditioning.text_encoder must be set when conditioning.type='text'.")


##############################################################
# Shared helpers used by both stage2/engine
##############################################################
def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    if isinstance(model_conds['context'], dict):
        from stage2 import nwm_cond
        return nwm_cond.apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob)
    mask = torch.rand(model_conds['context'].shape[0], device=model_conds['context'].device) < cfg_dropout_prob
    return {
        k: torch.where(mask.view(-1, *([1]*(v.ndim-1))), model_conds_null[k], v) if v is not None else None
        for k, v in model_conds.items()
    }, mask


def get_null_cond(text_encoder,conditioning_type, num_classes, batch_size, device):
    if conditioning_type == "text":
        _null_context, _null_attn_mask = encode_text(text_encoder, [""])
    else:
        _null_context, _null_attn_mask = torch.tensor([num_classes], device=device), None
    rtn = dict(context=_null_context, attn_mask=_null_attn_mask)
    rtn = {k: v.expand(batch_size, *v.shape[1:]) if v is not None else None for k, v in rtn.items()}
    return rtn


def setup_text_encoder(config, rank, device):
    """Build text encoder if conditioning.type == 'text', else return None.

    Side effect: sets config.conditioning.context_dim from the encoder's feature_dim.
    """
    if config.conditioning.type != "text":
        return None
    with main_process_first(rank):
        text_encoder = TextEncoder(**dataclasses.asdict(config.conditioning.text_encoder)).to(device)
    config.conditioning.context_dim = text_encoder.feature_dim
    return text_encoder


def encode_text(text_encoder, y):
    """Encode text conditions. Returns (encoder_hidden_states, encoder_attention_mask)."""
    with torch.no_grad():
        enc_out = text_encoder(y)
        return enc_out["tokens"], enc_out["attention_mask"]


def get_fixed_viz_batch_conditions(viz_fixed, y, condition_type, text_encoder, device):
    """Get fixed conditions for the first batch for consistent visualization."""
    if viz_fixed['context'] is not None:
        return viz_fixed
    n = viz_fixed['zs'].shape[0]
    if condition_type == "label":
        viz_fixed['context'] = y[:n].clone().to(device)
    else:
        with torch.no_grad():
            enc_out = text_encoder(y[:n])
            viz_fixed['context'] = enc_out["tokens"]
            viz_fixed['attn_mask'] = enc_out["attention_mask"]
    return viz_fixed


def sample_and_decode(
    zs, context, attn_mask,
    eval_sampler, model_fn, sample_model_kwargs, rae,
    use_guidance, condition_type, text_encoder, num_classes, device, autocast_kwargs,
):
    """Generate and decode samples, handling guidance doubling."""
    n = zs.shape[0]
    if use_guidance:
        zs = torch.cat([zs, zs], dim=0)
        if isinstance(context, dict):
            null = {k: torch.zeros_like(v) for k, v in context.items()}
            context = {k: torch.cat([context[k], null[k]], dim=0) for k in context}
            attn_mask = None
        else:
            if condition_type == "text":
                context_null, attn_mask_null = encode_text(text_encoder, [""] * n)
            else:
                context_null = torch.full((n,), num_classes, device=device)
                attn_mask_null = None
            context = torch.cat([context, context_null], dim=0)
            if attn_mask is not None and attn_mask_null is not None:
                attn_mask = torch.cat([attn_mask, attn_mask_null], dim=0)

    kwargs = deepcopy(sample_model_kwargs)
    kwargs.update(context=context, attn_mask=attn_mask)
    with autocast(**autocast_kwargs):
        samples = eval_sampler(zs, model_fn, **kwargs)[-1]
        if use_guidance:
            samples = samples.chunk(2, dim=0)[0]
    return rae.decode(samples).cpu().float()
