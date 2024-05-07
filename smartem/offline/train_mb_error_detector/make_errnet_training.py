import argparse

parser = argparse.ArgumentParser(
    description="Generates an ERRNET training dataset given a membrane predictor, and short dwell time images with ground truth membrane masks."
)
parser.add_argument("--model_weights", help="model weights of EM2MB network")
parser.add_argument(
    "--in_dataset_h5",
    help="dataset with short dwell time images and ground truth masks",
)
parser.add_argument("--out_dataset_h5", help="path of dataset to generate")
parser.add_argument(
    "--base_dwt", type=int, default=50, help="dwell time images to be used"
)
parser.add_argument(
    "--slowest_dwt", type=int, default=1200, help="slowest dwell time images to be used"
)
parser.add_argument(
    "--make_EM2Err",
    type=bool,
    default=False,
    help="Make dataset where x=<EM image> and y=<membrane prediction discrepancy from slowest dwell time image>",
)
args = parser.parse_args()

from smartem.offline.train_mb_error_detector.NNtools import UNet
from smartem.segmentation import segmenter
from tools import get_error_map, get_error_GT
import torch
import h5py
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
in_dataset_h5 = args.in_dataset_h5
out_dataset_h5 = args.out_dataset_h5
model_weights = args.model_weights
base_dwt = args.base_dwt
slowest_dwt = args.slowest_dwt
EM2err = args.make_EM2Err


# Make segmenter object which will perform prediction
net = UNet.UNet(n_channels=1, n_classes=2)
Iseg = segmenter.Segmenter(model_weights, device=device)
Iseg.set_model(model_class=net)

# Make predictions and compute errors
with h5py.File(in_dataset_h5, "r") as h5:

    regs = h5.attrs["regs"]
    data_to_save = {}
    out_regs = []
    for reg in tqdm(regs, desc="generating membrane predictions..."):
        hdt_im = h5[reg + "/" + str(slowest_dwt) + "/im"]
        _, hdt_mb_probs = Iseg.get_membranes(hdt_im, get_probs=True)
        hdt_mb_probs = (hdt_mb_probs * 255).astype(np.uint8)

        im = h5[reg + "/" + str(base_dwt) + "/im"]
        im = im[:]
        _, mb_probs = Iseg.get_membranes(im, get_probs=True)
        mb_probs = (mb_probs * 255).astype(np.uint8)

        # Generate error mask

        emap = get_error_GT(mb_probs, hdt_mb_probs)
        emap = (emap * 255).astype(np.uint8)

        # Organize data for saving later
        if EM2err:

            data_to_save[reg] = (im, emap)
        else:
            data_to_save[reg] = (mb_probs, emap)
        out_regs.append(reg)

with h5py.File(out_dataset_h5, "a") as h5:
    for reg in data_to_save.keys():
        mb_probs, labels = data_to_save[reg]
        g = h5.create_group(f"{reg}/{base_dwt}")
        g.create_dataset("im", data=mb_probs)
        g.create_dataset("mask", data=labels)

    dwts = [base_dwt]
    out_regs.sort()
    h5.attrs["dwts"] = dwts
    h5.attrs["regs"] = out_regs
    h5.attrs["H"] = mb_probs.shape[0]
    h5.attrs["W"] = mb_probs.shape[1]
