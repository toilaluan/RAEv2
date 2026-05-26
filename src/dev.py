from stage1.rae import RAE
from encoders.vision_encoder import VisionEncoder
from PIL import Image
import torch
from torchvision.transforms import PILToTensor, CenterCrop, Resize, ToPILImage
from transformers import AutoModel
import torch.nn as nn

from math import sqrt
### Original

device = "cuda"
rae = RAE(
    encoder_name="dinov2-vit-b",
    decoder_config_path="configs/decoder/ViTXL/config.json",
    pretrained_decoder_path="checkpoints/stage1/imagenet/dinov2b-k1/decoder.pt",
    normalization_stat_path="checkpoints/stage1/imagenet/dinov2b-k1/stats.pt",
    noise_tau=0.0,
).to(device, torch.bfloat16)

print(rae)


def test_recon(rae_model, input_path, output_path):
    img = Image.open(input_path).convert("RGB")

    img = Resize(224)(CenterCrop(224)(PILToTensor()(img)))
    img = img.unsqueeze(0).cuda().bfloat16()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        z = rae_model.encode(img)
        recon = rae_model.decode(z)
    print(recon.shape)

    recon = recon.clamp(0, 1)
    recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    recon_img = Image.fromarray(recon_np[0])
    recon_img.save(output_path)


test_recon(rae, "assets/samples/sample_2.png", "assets/sample_2_recon.jpg")

### Add new encoder


class AutoEncoderDino(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            "toilaluan/ae-dinov2-base", trust_remote_code=True
        )
        self.patch_size = 14
        self._embed_dim = self.model.backbone.hidden_size

    def preprocess(self, x):
        x = torch.nn.functional.interpolate(x, 224, mode="bicubic")
        return x

    def encode(self, x: torch.Tensor):
        # DINOv2 returns a dictionary with cls and patch tokens
        out = self.model(x, output_decoded_patches=False)
        return out.encoder_hidden_states  # B, num_prefix + num_square_patches_flattened

    def decode(self, encoder_hidden_states: torch.Tensor):
        prefix_tokens = encoder_hidden_states[:, : self.model.num_prefix_tokens]
        patch_tokens = encoder_hidden_states[:, self.model.num_prefix_tokens :]
        print(prefix_tokens.shape, patch_tokens.shape)
        out = self.model.decode(prefix_tokens, patch_tokens)
        return out


class AERAE(RAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = AutoEncoderDino()

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

        z = self.encoder.encode(x)

        if self.do_normalization:
            latent_mean = (
                self.latent_mean.to(z.device)
                if self.latent_mean is not None
                else torch.tensor(0.0, device=z.device)[None, None, :]
            )
            latent_var = (
                self.latent_var.to(z.device)
                if self.latent_var is not None
                else torch.tensor(1.0, device=z.device)[None, None, :]
            )
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = (
                self.latent_mean.to(z.device)
                if self.latent_mean is not None
                else torch.tensor(0.0, device=z.device)
            )
            latent_var = (
                self.latent_var.to(z.device)
                if self.latent_var is not None
                else torch.tensor(1.0, device=z.device)
            )
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        z_recon = self.encoder.decode(z)
        output = self.decoder(z_recon, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        x_rec = self.decode(z)
        if return_latent:
            return x_rec, z
        return x_rec


ae_rae = AERAE(
    encoder_name="dinov2-vit-b",
    decoder_config_path="configs/decoder/ViTXL/config.json",
    pretrained_decoder_path="checkpoints/stage1/imagenet/dinov2b-k1/decoder.pt",
    normalization_stat_path=None,
    noise_tau=0.0,
).cuda()

test_recon(ae_rae, "assets/samples/sample_2.png", "assets/sample_2_recon_ae.jpg")
