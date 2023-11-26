import connectomics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from connectomics.model.arch import UNet2D, UNetPlus2D
from torch import optim
from torch.utils.data import DataLoader, random_split
import os, sys, shutil

import data
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pickle
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import argparse


def train(num_epochs, batch_size, learning_rate, args, dwell_time_str="00025ns"):
    dwell_time_str = args.dwell_time_str
    data_direc = args.data_direc
    with open(
        "{}/global_{}/{}.pickle".format(data_direc, dwell_time_str, dwell_time_str),
        "rb",
    ) as handle:
        b = pickle.load(handle)

    print("Loading Data of dwell time", dwell_time_str)
    dataset = dataloading.MustExcludeDataset(b)

    # batch_size=5

    # 2. Split into train / validation partitions
    val_percent = 0.2
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # model = UNet2D(in_channel=1, out_channel=2)
    model = UNetPlus2D(in_channel=1, out_channel=2)
    model = model.to(memory_format=torch.channels_last)
    # model= nn.DataParallel(model,device_ids = [0])
    model = model.to(device=device)

    # learning_rate = learning_rate
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    epochs = num_epochs
    step = 0
    train_losses = []
    val_losses = []
    best_validation = 10000

    dir_checkpoint = "checkpoint_{}".format(dwell_time_str.strip("0"))

    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    for epoch in range(epochs):
        print("Epoch", epoch + 1, "/", epochs)
        model.train()
        train_loss_avg = 0

        for batch in tqdm(train_loader):
            frame, mask = batch["image"], batch["mask"]
            frame = frame.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.long)
            # print(frame.shape)
            # print(mask.shape)
            # frame = frame.unsqueeze(1)
            pred = model(frame)
            loss = criterion(pred, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_avg += loss.item()
            step += 1
            # update()
        train_loss_avg /= len(train_loader)
        print("train_losses = {}".format(train_loss_avg))
        train_losses.append(train_loss_avg)
        model.eval()
        val_loss_avg = 0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                frame, mask = batch["image"], batch["mask"]
                frame = frame.to(device=device, dtype=torch.float32)
                mask = mask.to(device=device, dtype=torch.long)

                pred = model(frame)
                loss = criterion(pred, mask)
                # loss += dice_loss(F.sigmoid(pred), mask.float(), multiclass=False)
                val_loss_avg += loss.item()
                n_val += 1
        val_loss_avg = val_loss_avg / n_val
        print("val_losses = {}".format(val_loss_avg))
        # t_vals.append(step)
        val_losses.append(val_loss_avg)
        if val_loss_avg < best_validation:
            print(
                "New best validation loss, previous best -> ",
                best_validation,
                " new best -> ",
                val_loss_avg,
            )
            best_validation = val_loss_avg
            torch.save(
                model.state_dict(), os.path.join(dir_checkpoint, "checkpoint_b.pth")
            )


def main():
    # print("Training")

    parser = argparse.ArgumentParser(
        description="Parser for dwell time, epochs, learning rate, and batch size."
    )

    parser.add_argument(
        "--dwell-time",
        type=str,
        default="00025ns",
        help="String for dwell time of the microscope (default: 00025ns)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs to train the model for (default: 25)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer during training (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of samples in each batch (default: 32)",
    )
    parser.add_argument(
        "--data-direc",
        type=str,
        default=None,
        help="data directory where the patches are stored",
    )

    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.learning_rate, args.dwell_time)


if __name__ == "__main__":
    main()
