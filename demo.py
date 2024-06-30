import numpy as np
import torch
import math
import cv2
from torchvision.utils import save_image
from models import UNet
from diffusers.models import AutoencoderKL


if __name__ == '__main__':
    weights = "workspace/rf_two_face_latent/checkpoints/best.pth"

    batch_size = 4
    sample_step = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # vae = None
    vae = AutoencoderKL.from_pretrained("E:/diffusion_models_zoo/stable-diffusion-v1-5", subfolder="vae")
    vae.to(device)
    vae = vae.half()

    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                 num_res_blocks=2, dropout=0.1, in_channels=4)
    model.to(device)
    model = model.half()
    checkpoint = torch.load(weights)
    # if isinstance(checkpoint, dict):
    #     checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)

    noise = torch.randn((batch_size, 4, 32, 32), dtype=torch.float16).to(device)
    pred = model.sample_ode(noise, sample_step)

    if vae is not None:
        with torch.no_grad():
            pred = vae.decode(pred / vae.config.scaling_factor, return_dict=False)[0]
    save_image(pred, f"check/rf2_196_s{sample_step:04d}.png", nrow=math.ceil(batch_size ** 0.5), normalize=True, value_range=(-1, 1))
    # pred = pred.cpu()
    # pred = torch.permute(pred, (0, 2, 3, 1))
    # images = np.clip(((pred / 2 + 0.5) * 255).numpy(), 0, 255).astype(np.uint8)
    # images = images[..., ::-1].copy()
    # for i, im in enumerate(images):
    #     cv2.imwrite(f"check/{i}.png", im)