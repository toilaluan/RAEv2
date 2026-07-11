from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from configs.stage2 import Stage2Config
from stage1.huggingface_rae import HuggingFaceCompressedRAE
from stage2.utils import validate_rae_latent_contract


class _FakeBaseRAE(nn.Module):
    pass


class _FakeCompressedRAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_size=4,
            patch_size=1,
            pool_size=2,
            hidden_size=2,
        )
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.weight = nn.Parameter(torch.ones(()))
        self.last_decoded = None

    def encode(self, images):
        values = torch.arange(8, device=images.device, dtype=torch.float32)
        return values.reshape(1, 2, 2, 2).expand(images.shape[0], -1, -1, -1)

    def decode(self, latents):
        self.last_decoded = latents.detach().clone()
        return latents


class HuggingFaceCompressedRAETest(unittest.TestCase):
    def _build(self, stats_path=None):
        base = _FakeBaseRAE()
        compressed = _FakeCompressedRAE()
        calls = []

        def load(name, **kwargs):
            calls.append((name, kwargs))
            return base if len(calls) == 1 else compressed

        mocked = patch(
            "stage1.huggingface_rae.AutoModel.from_pretrained",
            side_effect=load,
        )
        with mocked:
            model = HuggingFaceCompressedRAE(
                base_model_name_or_path="base",
                compressor_model_name_or_path="compressor",
                base_revision="base-sha",
                compressor_revision="compressor-sha",
                normalization_stat_path=stats_path,
                dtype="bfloat16",
            )
        return model, base, compressed, calls

    def test_loading_and_normalized_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / "stats.pt"
            mean = torch.full((2, 2, 2), 2.0)
            var = torch.full((2, 2, 2), 4.0)
            torch.save({"mean": mean, "var": var}, stats_path)

            model, base, compressed, calls = self._build(str(stats_path))
            raw = model.encode_raw(torch.zeros(3, 3, 4, 4))
            normalized = model.encode(torch.zeros(3, 3, 4, 4))
            expected = (raw - mean) / torch.sqrt(var + model.eps)
            self.assertTrue(torch.allclose(normalized, expected))

            reconstruction = model.decode(normalized)
            self.assertTrue(torch.allclose(compressed.last_decoded, raw))
            self.assertTrue(torch.allclose(reconstruction, raw))
            self.assertEqual(model.latent_shape, (2, 2, 2))
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))

            self.assertEqual(calls[0][0], "base")
            self.assertEqual(calls[0][1]["revision"], "base-sha")
            self.assertEqual(calls[0][1]["dtype"], torch.bfloat16)
            self.assertTrue(calls[0][1]["trust_remote_code"])
            self.assertEqual(calls[1][0], "compressor")
            self.assertIs(calls[1][1]["rae"], base)
            self.assertEqual(calls[1][1]["revision"], "compressor-sha")

    def test_rejects_incompatible_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / "stats.pt"
            torch.save(
                {"mean": torch.zeros(2, 4, 4), "var": torch.ones(2, 4, 4)},
                stats_path,
            )
            with self.assertRaisesRegex(ValueError, "Do not reuse 16x16"):
                self._build(str(stats_path))

    def test_latent_contract_validation(self):
        model, _, _, _ = self._build()
        config = Stage2Config()
        config.misc.latent_size = [2, 2, 2]
        config.stage_2.params = {"in_channels": 2, "input_size": 2}
        validate_rae_latent_contract(model, config)

        config.stage_2.params["input_size"] = 4
        with self.assertRaisesRegex(ValueError, "input_size"):
            validate_rae_latent_contract(model, config)


if __name__ == "__main__":
    unittest.main()
