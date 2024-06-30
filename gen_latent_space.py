import torch.cuda
from diffusers.models import AutoencoderKL
import os
from glob import glob
from torchvision.transforms import transforms
from PIL import Image


@torch.no_grad()
def gen_latent_space(src, dst):
    trans = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    for cls_name in os.listdir(src):
        cls_root = os.path.join(src, cls_name)
        dst_cls_root = os.path.join(dst, cls_name)
        os.makedirs(dst_cls_root, exist_ok=True)
        for i, filename in enumerate(glob(os.path.join(cls_root, "*"))):
            im = Image.open(filename)
            x = trans(im)[None].to(device)
            latents = vae.encode(x).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            filename_dst = os.path.join(dst_cls_root, f"{i:02d}.pth")
            torch.save(latents, filename_dst)
            print(f"{filename}: {latents.shape} done")
    print("done")

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("E:/diffusion_models_zoo/stable-diffusion-v1-5", subfolder="vae")
vae.to(device)
gen_latent_space(r"D:\temp_data\lfw-align-128", r"D:\temp_data\lfw-align-128_latent")