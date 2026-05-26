from stage1.rae import RAE
from PIL import Image
import torch
from torchvision.transforms import PILToTensor, CenterCrop, Resize, ToPILImage
from transformers import AutoModel
import torch.nn as nn

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


class FVRAE(RAE):
    def __init__(self, finevit, fv_mean=None, fv_var=None, **kwargs):
        super().__init__(**kwargs)
        self.finevit = finevit
        self.finevit_latent_mean = fv_mean
        self.finevit_latent_var = fv_var

    def encode(self, x):
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
        x = torch.nn.functional.interpolate(
            x, 224 * (self.resolution // 256), mode="bicubic"
        )
        out = self.finevit.encode(x)
        z = out.encoder_hidden_states

        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.finevit_latent_mean is not None:
            latent_mean = (
                self.finevit_latent_mean.to(z.device)
                if self.finevit_latent_mean is not None
                else 0
            )
            latent_var = (
                self.finevit_latent_var.to(z.device)
                if self.finevit_latent_var is not None
                else 1
            )
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z):
        if self.finevit_latent_mean is not None:
            latent_mean = (
                self.finevit_latent_mean.to(z.device)
                if self.finevit_latent_mean is not None
                else 0
            )
            latent_var = (
                self.finevit_latent_var.to(z.device)
                if self.finevit_latent_var is not None
                else 1
            )
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean

        prefix_tokens = z[:, :self.finevit.num_prefix_tokens, :]
        patch_tokens = z[:, self.finevit.num_prefix_tokens :, :]
        z_recon = self.finevit.decode(prefix_tokens, patch_tokens)
        output = self.decoder(z_recon, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)


test_recon(rae, "assets/samples/sample_2.png", "assets/sample_2_recon.jpg")

finevit = AutoModel.from_pretrained("toilaluan/ae-dino-2b", trust_remote_code=True)
finevit = finevit.to(device, torch.bfloat16)


rae_w_fv = (
    FVRAE(
        finevit,
        encoder_name="dinov2-vit-b",
        decoder_config_path="configs/decoder/ViTXL/config.json",
        pretrained_decoder_path="checkpoints/stage1/imagenet/dinov2b-k1/decoder.pt",
        normalization_stat_path="checkpoints/stage1/imagenet/dinov2b-k1/stats.pt",
        noise_tau=0.0,
    )
    .cuda()
    .bfloat16()
)

test_recon(rae_w_fv, "assets/samples/sample_2.png", "assets/sample_2_recon_fv.jpg")
