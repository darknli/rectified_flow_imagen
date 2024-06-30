import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch_frame import AccelerateTrainer
from torch_frame.hooks import LoggerHook, CheckpointerHook
from dataset import RectifiedFlowOneDataset, RectifiedFlowOneLatentDataset, RectifiedFlowTwoLatentDataset
from models import UNet


def train_rf1():
    num_epochs = 100
    lr = 1e-4
    num_workers = 8
    batch_size = 256

    # trainset = RectifiedFlowOneDataset("./dataset", True)
    trainset = RectifiedFlowOneLatentDataset(r"D:\temp_data\lfw-align-128_latent")

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              persistent_workers=True,
                              # collate_fn=collect_fn
                              )

    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                 num_res_blocks=2, dropout=0.1, in_channels=4)
    optimizer = AdamW(model.parameters(), lr)

    hooks = [
        CheckpointerHook(save_metric="total_loss", max_first=False),
        LoggerHook(),
    ]

    trainer = AccelerateTrainer(model, optimizer, "constant", train_loader, num_epochs,
                                hooks=hooks, mixed_precision="fp16", work_dir="workspace/rf_one_face_latent")
    trainer.train()


def train_rf2():
    num_epochs = 200
    lr = 1e-4
    num_workers = 8
    batch_size = 256

    # trainset = RectifiedFlowOneDataset("./dataset", True)
    trainset = RectifiedFlowTwoLatentDataset(r"D:\temp_data\lfw-align-128_pair_rf1")

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              persistent_workers=True,
                              # collate_fn=collect_fn
                              )

    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                 num_res_blocks=2, dropout=0.1, in_channels=4)
    optimizer = AdamW(model.parameters(), lr)

    hooks = [
        CheckpointerHook(save_metric="total_loss", max_first=False),
        LoggerHook(),
    ]

    trainer = AccelerateTrainer(model, optimizer, "constant", train_loader, num_epochs,
                                hooks=hooks, mixed_precision="fp16", work_dir="workspace/rf_two_face_latent")
    trainer.train()


if __name__ == '__main__':
    train_rf2()
