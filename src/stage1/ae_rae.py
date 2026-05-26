from math import prod
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel

from .rae import _load_decoder, _load_normalization_stats


class AERAE(nn.Module):
    """RAE wrapper for AE-DINO latents with prefix + patch token sequences."""

    def __init__(
        self,
        encoder_name: str = "toilaluan/ae-dinov2-base",
        resolution: int = 256,
        ae_input_size: int = 224,
        decoder_config_path: str = "vit_mae-base",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.0,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        num_prefix_tokens: int = 9,
        patch_grid_size: Sequence[int] = (8, 8),
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.resolution = resolution
        self.ae_input_size = ae_input_size
        self.noise_tau = noise_tau
        self.eps = eps
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_grid_size = tuple(patch_grid_size)
        self.num_patch_tokens = prod(self.patch_grid_size)
        self.num_tokens = self.num_prefix_tokens + self.num_patch_tokens

        self.encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=trust_remote_code)
        self.encoder_patch_size = getattr(self.encoder, "patch_size", 14)
        self.latent_dim = self._infer_latent_dim()
        self.base_patches = (resolution // decoder_patch_size) ** 2

        self.decoder = _load_decoder(
            decoder_config_path,
            self.latent_dim,
            decoder_patch_size,
            self.base_patches,
            pretrained_decoder_path,
        )
        self.latent_mean, self.latent_var, self.do_normalization = _load_normalization_stats(normalization_stat_path)
        print(
            f"AERAE: encoder={encoder_name}, resolution={resolution}, "
            f"tokens={self.num_tokens} ({self.num_prefix_tokens}+{self.patch_grid_size[0]}x{self.patch_grid_size[1]}), "
            f"hidden_size={self.latent_dim}"
        )

    def _infer_latent_dim(self) -> int:
        backbone = getattr(self.encoder, "backbone", None)
        hidden_size = getattr(backbone, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Could not infer AE-DINO hidden size from backbone.hidden_size or config.hidden_size.")
        return int(hidden_size)

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        return x + noise_sigma * torch.randn_like(x)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            x,
            size=(self.ae_input_size, self.ae_input_size),
            mode="bicubic",
            align_corners=False,
        )

    def _validate_latents(self, z: torch.Tensor) -> None:
        if z.ndim != 3:
            raise ValueError(f"AERAE expected sequence latents with shape [B, N, C], got {tuple(z.shape)}")
        if z.shape[1] != self.num_tokens or z.shape[2] != self.latent_dim:
            raise ValueError(
                f"AERAE expected latent shape [B, {self.num_tokens}, {self.latent_dim}], got {tuple(z.shape)}"
            )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() <= 1.0:
            x = x * 255.0
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode="bicubic",
                align_corners=False,
            )

        x = self.preprocess(x)
        out = self.encoder(x, output_decoded_patches=False)
        z = out.encoder_hidden_states
        self._validate_latents(z)

        if self.training and self.noise_tau > 0:
            z = self.noising(z)

        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self._validate_latents(z)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean

        prefix_tokens = z[:, : self.num_prefix_tokens]
        patch_tokens = z[:, self.num_prefix_tokens :]
        decoded_patches = self.encoder.decode(prefix_tokens, patch_tokens)
        output = self.decoder(decoded_patches, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        x_rec = self.decode(z)
        if return_latent:
            return x_rec, z
        return x_rec
