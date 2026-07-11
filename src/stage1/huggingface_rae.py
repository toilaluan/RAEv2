"""Hugging Face backed compact RAE compatible with the stage-2 interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel


_DTYPES = {
    "auto": "auto",
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(dtype: str | torch.dtype) -> str | torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[dtype.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_DTYPES))
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of: {supported}") from exc


class HuggingFaceCompressedRAE(nn.Module):
    """Compose a self-contained Hugging Face RAE with its spatial compressor.

    The Hugging Face checkpoints contain raw, unnormalized stage-1 features.
    This wrapper applies compact-latent normalization at the boundary expected
    by stage-2 training and reverses it before decoding.
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        compressor_model_name_or_path: str,
        base_revision: Optional[str] = None,
        compressor_revision: Optional[str] = None,
        normalization_stat_path: Optional[str] = None,
        dtype: str | torch.dtype = "auto",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")

        load_kwargs = {
            "trust_remote_code": True,
            "dtype": _resolve_dtype(dtype),
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        base_kwargs = dict(load_kwargs)
        compressor_kwargs = dict(load_kwargs)
        if base_revision is not None:
            base_kwargs["revision"] = base_revision
        if compressor_revision is not None:
            compressor_kwargs["revision"] = compressor_revision

        base_rae = AutoModel.from_pretrained(base_model_name_or_path, **base_kwargs)
        self.model = AutoModel.from_pretrained(
            compressor_model_name_or_path,
            rae=base_rae,
            **compressor_kwargs,
        )
        self.model.requires_grad_(False)
        self.model.eval()

        config = self.model.config
        self.resolution = int(config.image_size)
        self.latent_dim = int(config.hidden_size)
        self.encoder_patch_size = int(config.patch_size)
        pool_size = int(config.pool_size)
        encoder_grid = self.resolution // self.encoder_patch_size
        compact_grid = encoder_grid // pool_size
        self._latent_shape = (self.latent_dim, compact_grid, compact_grid)
        self.base_patches = compact_grid**2
        self.eps = eps
        self.base_model_name_or_path = base_model_name_or_path
        self.compressor_model_name_or_path = compressor_model_name_or_path
        self.base_revision = base_revision
        self.compressor_revision = compressor_revision

        mean, var = self._load_stats(normalization_stat_path)
        self.register_buffer("latent_mean", mean, persistent=False)
        self.register_buffer("latent_var", var, persistent=False)
        self.do_normalization = mean is not None

        print(
            "HuggingFaceCompressedRAE: "
            f"base={base_model_name_or_path}, compressor={compressor_model_name_or_path}, "
            f"latent_shape={self.latent_shape}, normalized={self.do_normalization}"
        )

    @property
    def encoder(self) -> nn.Module:
        return self.model.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.model.decoder

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self._latent_shape

    def _load_stats(
        self, path: Optional[str]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if path is None:
            return None, None
        stats_path = Path(path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Compact RAE normalization stats not found: {path}")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        mean, var = stats.get("mean"), stats.get("var")
        if mean is None or var is None:
            raise ValueError(f"Stats file {path} must contain both 'mean' and 'var'")
        expected = self.latent_shape
        if tuple(mean.shape) != expected or tuple(var.shape) != expected:
            raise ValueError(
                f"Stats shape mismatch for compact RAE: expected {expected}, "
                f"got mean={tuple(mean.shape)}, var={tuple(var.shape)}. "
                "Do not reuse 16x16 RAE statistics."
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
            raise ValueError(f"Stats file {path} contains non-finite values")
        if (var < 0).any():
            raise ValueError(f"Stats file {path} contains negative variance")
        expected_metadata = {
            "base_model_name_or_path": self.base_model_name_or_path,
            "compressor_model_name_or_path": self.compressor_model_name_or_path,
            "base_revision": self.base_revision,
            "compressor_revision": self.compressor_revision,
        }
        for key, expected_value in expected_metadata.items():
            saved_value = stats.get(key)
            if saved_value is not None and expected_value is not None and saved_value != expected_value:
                raise ValueError(
                    f"Stats file {path} was computed for {key}={saved_value!r}, "
                    f"but the configured model uses {expected_value!r}"
                )
        print(f"Loaded compact RAE normalization stats from {path}")
        return mean.float(), var.float()

    @torch.no_grad()
    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(x)
        if tuple(z.shape[1:]) != self.latent_shape:
            raise RuntimeError(
                f"Hugging Face compressor returned {tuple(z.shape[1:])}; "
                f"expected {self.latent_shape}"
            )
        return z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_raw(x)
        if self.do_normalization:
            z = (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if tuple(z.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Expected compact latents shaped [B, {self.latent_shape}], "
                f"got {tuple(z.shape)}"
            )
        if self.do_normalization:
            z = z * torch.sqrt(self.latent_var + self.eps) + self.latent_mean
        return self.model.decode(z)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        reconstruction = self.decode(z)
        if return_latent:
            return reconstruction, z
        return reconstruction
