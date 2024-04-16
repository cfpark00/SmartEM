import argparse

parser = argparse.ArgumentParser(description='Generates an ERRNET training dataset given a membrane predictor, and short dwell time images with ground truth membrane masks.')
parser.add_argument('--model_weights', help='model weights of EM2MB network')
parser.add_argument('--in_dataset_h5', help='dataset with short dwell time images and ground truth masks')
parser.add_argument('--out_dataset_h5', help='path of dataset to generate')
parser.add_argument('-dwt',type=int,default=50, help='dwell time images to be used')
parser.add_argument('--slowest_dwt',type=int,default=1200, help='slowest dwell time images to be used')
parser.add_argument('--make_EM2Err',type=bool,default=False, help='make EM images and error mask dataset')
args=parser.parse_args()
import sys, os

try:
    # from smartem.online.models import UNet
    from smartem.offline.train_mb_error_detector.NNtools import UNet
    from smartem.segmentation import segmenter
except:

    sys.path.append('/home/ssawmya-local/FM_work/SmartEM/')
    sys.path.append('/home/ssawmya-local/FM_work/SmartEM/smartem/segmentation')
    # from smartem.online.models import UNet
    from smartem.offline.train_mb_error_detector.NNtools import UNet
    from smartem.segmentation import segmenter




from tools import get_error_map, get_error_GT
import torch
import h5py
from tqdm import tqdm
import numpy as np

device="cuda" if torch.cuda.is_available() else "cpu"
in_dataset_h5=args.in_dataset_h5
out_dataset_h5=args.out_dataset_h5
model_weights=args.model_weights
dwt=args.dwt
slowest_dwt = args.slowest_dwt
EM2err = args.make_EM2Err


# Make segmenter object which will perform prediction
net = UNet.UNet(n_channels=1,n_classes=2)
Iseg = segmenter.Segmenter(model_weights, device=device)
Iseg.set_model(model_class=net)

# Make predictions and compute errors
with h5py.File(in_dataset_h5, 'r') as h5:


    regs=h5.attrs["regs"]
    data_to_save = {}
    out_regs = []

    hdt_im, _ = h5[regs[0]+"/"+str(slowest_dwt)+"/im"], h5[regs[0]+"/"+str(slowest_dwt)+"/mask"]
    _, hdt_mb_probs = Iseg.get_membranes(hdt_im, get_probs=True)
    # print(np.unique(hdt_mb_probs))
    # make hdt_mb_probs uint8
    hdt_mb_probs = (hdt_mb_probs*255).astype(np.uint8)

    
    for reg in tqdm(regs, desc="generating membrane predictions..."):
        im,mask=h5[reg+"/"+str(dwt)+"/im"],h5[reg+"/"+str(dwt)+"/mask"]
        im = im[:]
        # Make predictions
        # mask=Iseg.preprocess(mask)
        # mask = np.squeeze(mask)
        _, mb_probs = Iseg.get_membranes(im, get_probs=True)
        # mask = (mask*255).astype(np.uint8)
        mb_probs = (mb_probs*255).astype(np.uint8)
        #print(f"predicted probs shape: {mb_probs.shape}, {mb_probs.dtype} {np.amin(mb_probs)}-{np.amax(mb_probs)} w/median {np.median(mb_probs)}, sum {np.sum(mask)}")
        #print(f"mask shape: {mask.shape}, {mask.dtype} {np.amin(mask)}-{np.amax(mask)} w/median {np.median(mask)}, sum {np.sum(mask)}")

        # Generate error mask

        emap = get_error_GT(mb_probs, hdt_mb_probs)
        emap = (emap*255).astype(np.uint8)

        # print(f"MB predictions shape: {mb_probs.shape}, {mb_probs.dtype} {np.amin(mb_probs)}-{np.amax(mb_probs)} w/median {np.median(mb_probs)}, sum {np.sum(mb_probs)}")
        # print(f"error labels shape: {emap.shape}, {emap.dtype} {np.amin(emap)}-{np.amax(emap)} w/median {np.median(emap)}, sum {np.sum(emap)}")

        # Organize data for saving later
        if EM2err:

            data_to_save[reg] = (im, emap)
        else:
            data_to_save[reg] = (mb_probs, emap)
        out_regs.append(reg)
        # break
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
