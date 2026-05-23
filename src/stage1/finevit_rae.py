from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .rae import RAE, _load_normalization_stats


class FineViTRAE(RAE):
    """RAE wrapper that uses FineViT bottleneck tokens as the stage-2 latent."""

    def __init__(
        self,
        finevit_model_name: str = "toilaluan/aabb",
        finevit_normalization_stat_path: Optional[str] = None,
        finevit_target_seq_length: int = 256,
        finevit_trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.finevit = AutoModel.from_pretrained(
            finevit_model_name,
            trust_remote_code=finevit_trust_remote_code,
        )
        self.finevit_target_seq_length = finevit_target_seq_length
        self.finevit_latent_mean, self.finevit_latent_var, self.do_finevit_normalization = (
            _load_normalization_stats(finevit_normalization_stat_path)
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

        x = self.encoder.preprocess(x)
        z = self.finevit(x, only_latents=True)

        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.do_finevit_normalization:
            latent_mean = self.finevit_latent_mean.to(z.device) if self.finevit_latent_mean is not None else 0
            latent_var = self.finevit_latent_var.to(z.device) if self.finevit_latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"FineViTRAE.decode expects 1D token latents [B, L, C], got shape {tuple(z.shape)}")
        if self.do_finevit_normalization:
            latent_mean = self.finevit_latent_mean.to(z.device) if self.finevit_latent_mean is not None else 0
            latent_var = self.finevit_latent_var.to(z.device) if self.finevit_latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean

        z_recon = self.finevit.patch_decoder(z, target_seq_length=self.finevit_target_seq_length)
        output = self.decoder(z_recon, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)
