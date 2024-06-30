import glob
import os

import numpy as np
import torch
import cv2
import tqdm
from torchvision.utils import save_image
from models import UNet
from diffusers.models import AutoencoderKL
from threading import Thread


def write(noise, batch_pred):
    noise, batch_pred = noise.cpu(), batch_pred.cpu()
    global i, pbar
    for n, p in zip(noise, batch_pred):
        # torch.save([n, p], os.path.join(dst, f"{i}.pth"))
        np.save(os.path.join(dst, f"{i}.npy"), np.stack([n, p]))
        i += 1
        pbar.update(1)
    print("线程结束")


if __name__ == '__main__':
    weights = "workspace/rf_one_face_latent/checkpoints/best.pth"
    dst = r"D:\temp_data\lfw-align-128_pair_rf1\one"

    batch_size = 512
    num_iter = 40000 // batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # vae = None
    vae = AutoencoderKL.from_pretrained("E:/diffusion_models_zoo/stable-diffusion-v1-5", subfolder="vae")
    vae.to(device)

    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                 num_res_blocks=2, dropout=0.1, in_channels=4)
    model.to(device)
    model = model.half()
    checkpoint = torch.load(weights)
    # if isinstance(checkpoint, dict):
    #     checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)

    os.makedirs(dst, exist_ok=True)

    i = len(glob.glob(os.path.join(dst, "*.pth")))
    print(i)
    pbar = tqdm.tqdm(total=num_iter * batch_size - i)
    for _ in range(num_iter):
        noise = torch.randn((batch_size, 4, 32, 32), dtype=torch.float16).to(device)
        batch_pred = model.sample_ode(noise)
        Thread(target=write, args=(noise, batch_pred)).start()
        # noise, batch_pred = noise.cpu(), batch_pred.cpu()
        # for n, p in zip(noise, batch_pred):
        #     # np.save(os.path.join(dst, f"{i}_noise.pth"), )
        #     torch.save([n, p], os.path.join(dst, f"{i}.pth"))
        #     i += 1
        #     pbar.update(1)
    pbar.close()

