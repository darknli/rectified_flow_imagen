import glob
import os.path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import torch


class RectifiedFlowOneDataset(Dataset):
    def __init__(self, root, train, ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.dataset = CIFAR10(root=root, train=train, download=True, transform=transform)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        noise = torch.randn_like(image)
        item = {
            "z_0": noise,
            "z_1": image
        }
        return item

    def __len__(self):
        return len(self.dataset)


class RectifiedFlowOneLatentDataset(Dataset):
    def __init__(self, root):
        self.dataset = glob.glob(os.path.join(root, "*", "*.pth"))

    def __getitem__(self, idx):
        filename = self.dataset[idx]
        image = torch.load(filename)[0].detach().cpu()
        noise = torch.randn_like(image)
        item = {
            "z_0": noise,
            "z_1": image
        }
        return item

    def __len__(self):
        return len(self.dataset)


class RectifiedFlowTwoLatentDataset(Dataset):
    def __init__(self, root):
        self.dataset = glob.glob(os.path.join(root, "*", "*.npy"))

    def __getitem__(self, idx):
        filename = self.dataset[idx]
        if filename.endswith(".pth"):
            noise, image = torch.load(filename)
        else:
            noise, image = np.load(filename)
            noise = torch.from_numpy(noise)
            image = torch.from_numpy(image)
        item = {
            "z_0": noise,
            "z_1": image
        }
        return item

    def __len__(self):
        return len(self.dataset)