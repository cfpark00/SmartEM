import argparse

parser = argparse.ArgumentParser(description='Generates an ERRNET training dataset given a membrane predictor, and short dwell time images with ground truth membrane masks.')
parser.add_argument('model_weights', help='model weights of EM2MB network')
parser.add_argument('in_dataset_h5', help='dataset with short dwell time images and ground truth masks')
parser.add_argument('out_dataset_h5', help='path of dataset to generate')
parser.add_argument('-dwt',type=int,default=50, help='dwell time images to be used')
args=parser.parse_args()

from FM_work.SmartEM.smartem.segmentation import UNet
from tools import get_error_map, get_error_GT
from smartem.segmentation import segmenter
import torch
import h5py
from tqdm import tqdm
import numpy as np

device="cuda" if torch.cuda.is_available() else "cpu"
in_dataset_h5=args.in_dataset_h5
out_dataset_h5=args.out_dataset_h5
model_weights=args.model_weights
dwt=args.dwt

# Make segmenter object which will perform prediction
net = UNet.UNet(n_channels=1,n_classes=2)
Iseg = segmenter.Segmenter(model_weights, device=device)
Iseg.set_model(model_class=net)

# Make predictions and compute errors
with h5py.File(in_dataset_h5, 'r') as h5:
    regs=h5.attrs["regs"]
    data_to_save = {}
    out_regs = []
    for reg in tqdm(regs, desc="generating membrane predictions..."):
        im,mask=h5[reg+"/"+str(dwt)+"/im"],h5[reg+"/"+str(dwt)+"/mask"]

        # print(f"image shape: {im.shape}, {im.dtype} {np.amin(im)}-{np.amax(im)} w/median {np.median(im)}, sum {np.sum(im)}")
        # print(f"mask shape: {mask.shape}, {mask.dtype} {np.amin(mask)}-{np.amax(mask)} w/median {np.median(mask)}, sum {np.sum(mask)}")

        # Make predictions
        mask=Iseg.preprocess(mask)
        mask = np.squeeze(mask)
        _, mb_probs = Iseg.get_membranes(im, get_probs=True)
        mask = (mask*255).astype(np.uint8)
        mb_probs = (mb_probs*255).astype(np.uint8)
        #print(f"predicted probs shape: {mb_probs.shape}, {mb_probs.dtype} {np.amin(mb_probs)}-{np.amax(mb_probs)} w/median {np.median(mb_probs)}, sum {np.sum(mask)}")
        #print(f"mask shape: {mask.shape}, {mask.dtype} {np.amin(mask)}-{np.amax(mask)} w/median {np.median(mask)}, sum {np.sum(mask)}")

        # Generate error mask
        labels = get_error_GT(mb_probs, mask)
        labels = labels.astype(np.uint8)*255

        # print(f"MB predictions shape: {mb_probs.shape}, {mb_probs.dtype} {np.amin(mb_probs)}-{np.amax(mb_probs)} w/median {np.median(mb_probs)}, sum {np.sum(mb_probs)}")
        # print(f"error labels shape: {labels.shape}, {labels.dtype} {np.amin(labels)}-{np.amax(labels)} w/median {np.median(labels)}, sum {np.sum(labels)}")

        # Organize data for saving later
        data_to_save[reg] = (mb_probs, labels)
        out_regs.append(reg)

# Save data
with h5py.File(out_dataset_h5, 'a') as h5:
    for reg in data_to_save.keys():
        mb_probs, labels = data_to_save[reg]
        g = h5.create_group(f"{reg}/{dwt}")
        g.create_dataset("im", data = mb_probs)
        g.create_dataset("mask", data = labels)

    
    dwts = [dwt]
    out_regs.sort()
    h5.attrs["dwts"] = dwts
    h5.attrs["regs"] = out_regs
    h5.attrs["H"] = mb_probs.shape[0]
    h5.attrs["W"] = mb_probs.shape[1]
