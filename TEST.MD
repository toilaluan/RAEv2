
```
hf download nyu-visionx/RAEv2-models --include stage1/imagenet/dinov2b-k1/* --local-dir checkpoints/
hf download nyu-visionx/RAEv2-models encoders/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --local-dir pretrained_models
hf download benjamin-paine/imagenet-1k-256x256 --local-dir ./data/imagenet_256 --repo-type dataset
```

```
PYTHONPATH=src uv run torchrun --nproc_per_node=1 scripts/stage1/compute_encoder_stats.py   \
--config configs/stage2/training/imagenet-finevit-dinov2b-k1.yaml   \
--use-hf-dataset   --hf-data-dir data/imagenet_256   --batch-size 256   \
--output-path pretrained_models/stage1/imagenet/finevit-dinov2b-k1/stats.pt --hf-split validation
```

```
uv run torchrun --nproc_per_node=1     src/train.py  \
--config configs/stage2/training/imagenet-finevit-dinov2b-k1.yaml    \
--results-dir ckpts/stage2     \
--precision bf16     \
--compile     \
--wandb
```