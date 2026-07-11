from __future__ import annotations

import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from configs.stage2 import ConditioningArchConfig
from stage2.models.DDT import DiTwDDTHeadIG


class CompactImageNetConfigTest(unittest.TestCase):
    def test_training_and_sampling_contracts(self):
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "configs/stage2/training/imagenet-dinov3l-k7-compressed.yaml",
            root / "configs/stage2/sampling/imagenet-dinov3l-k7-compressed.yaml",
        )
        for path in paths:
            with self.subTest(path=path):
                config = OmegaConf.load(path)
                self.assertEqual(config.stage_1.target, "stage1.HuggingFaceCompressedRAE")
                self.assertEqual(list(config.misc.latent_size), [1024, 8, 8])
                self.assertEqual(config.misc.time_dist_shift_dim, 65536)
                self.assertEqual(config.stage_2.params.input_size, 8)
                self.assertEqual(config.stage_2.params.in_channels, 1024)
                self.assertEqual(config.conditioning.type, "label")
                self.assertEqual(config.eval.datasets.imagenet.num_samples, 50000)
                self.assertEqual(
                    list(config.eval.datasets.imagenet.metrics),
                    ["fid", "inception_score"],
                )

    def test_ddt_accepts_an_eight_by_eight_latent_grid(self):
        model = DiTwDDTHeadIG(
            input_size=8,
            in_channels=4,
            patch_size=[1, 1],
            hidden_size=[32, 32],
            depth=[2, 1],
            num_heads=[4, 4],
            mlp_ratio=2.0,
            base_model_depth=1,
            num_classes=10,
            condition_type="label",
            context_dim=32,
            cond_arch=ConditioningArchConfig(num_t_tokens=2, num_c_tokens=2),
        )
        latents = torch.randn(2, 4, 8, 8)
        output, base_output = model(
            latents,
            torch.rand(2),
            context=torch.tensor([1, 2]),
            attn_mask=None,
        )
        self.assertEqual(tuple(output.shape), tuple(latents.shape))
        self.assertEqual(tuple(base_output.shape), tuple(latents.shape))


if __name__ == "__main__":
    unittest.main()
