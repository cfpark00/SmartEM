import os
import sys
import time

import numpy as np
import cv2
import glob
from PIL import Image
import scipy.ndimage as sim
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import shutil
import h5py
import torch
import torch.nn.functional as F

from smartem.timing import timing

clahe = cv2.createCLAHE(clipLimit=3).apply

def float_to_int(im, dtype=np.uint8):
    """
    Convert an image from float [0,1] to integer.

    Args:
    im: np.ndarray, image, float data type

    Returns:
    im: np.ndarray, image, integer data type
    """
    return np.clip(im * np.iinfo(dtype).max, 0, np.iinfo(dtype).max).astype(dtype)


def int_to_float(im, dtype=np.float32):
    """
    Convert an image from integer to float [0,1].

    Args:
    im: np.ndarray, image, integer data type

    Returns:
    im: np.ndarray, image, float data type
    """
    return im.astype(dtype) / np.iinfo(im.dtype).max


def get_logprob(logit, dim=1):
    """
    Get the log probability from the logit.

    Args:
    logit: torch.Tensor, logits
    dim: int, dimension to sum

    Returns:
    logprob: torch.Tensor, log probability
    """
    lse = torch.logsumexp(logit, dim=dim, keepdim=True)
    return logit - lse

@timing
def get_prob(image, net, return_dtype=np.uint8, check_nans=False):
    """
    Get the membrane probability map from the image using the net.

    Args:
    image: np.ndarray, image (W, H)
    net: torch.nn.Module, network
    return_dtype: np.dtype, return data type

    Returns:
    prob: np.ndarray, membrane probability map
    """

    image_dtype = image.dtype
    assert (
        image_dtype == np.uint8 or image_dtype == np.uint16 or image_dtype == np.float32
    )
    assert return_dtype == np.uint8 or return_dtype == np.float32

    if image_dtype == np.uint8 or image_dtype == np.uint16:
        image_torch = torch.tensor(int_to_float(image, dtype=np.float32))[None, None]
    else:
        image_torch = torch.tensor(image, dtype=torch.float32)[None, None]

    with torch.no_grad():
        image_torch = image_torch.to(device=next(net.parameters()).device, dtype=torch.float32)

        if check_nans:
            print(f"Image: {image_torch.shape}@{image_torch.dtype} w/nans: {torch.any(torch.isnan(image_torch))}")
            print(f"Model: {net.n_channels}channels-> {net.n_classes}classes ({net.training})")
            for name, param in net.named_parameters():
                if torch.isnan(param).any():
                    raise ValueError()         
            for name, param in net.named_buffers():
                if torch.isnan(param).any():
                    raise ValueError()

        with torch.autocast(device_type="cuda", enabled=True):
            mask_logits = net(
                image_torch
            )

        if check_nans and torch.any(torch.isnan(mask_logits)):
            debug_weights(net)
            debug_nan(net, image_torch)
            raise ValueError("Nan found")
        
        prob = (
            torch.exp(get_logprob(mask_logits))[0, 1].cpu().detach().numpy()
        )  # 1st channel for membrane

        if check_nans and np.any(np.isnan(prob.flatten())):
            raise ValueError("NaN found in probs")
    if return_dtype == np.uint8:
        return float_to_int(prob, dtype=return_dtype)
    else:
        return prob.astype(return_dtype)

def debug_weights(model):
    with torch.autocast(device_type="cuda"):
        nan_found = False
        for name, param in model.named_parameters():
            if 'weight' in name:
                if not torch.isfinite(param).all():
                    nan_count = torch.isnan(param).sum().item()
                    print(f"Layer: {name} contains Nans. No: {nan_count}")
                    nan_found = True
            if not nan_found:
                print("No Nans found in any weight matrices")

def debug_nan(model, input_tensor):
    with torch.autocast(device_type="cuda"):
        print("Debugging...")
        intermediate_activations = []
        for i, module in enumerate([model.inc, model.down1, model.down2, model.down3, model.down4]):
            input_tensor = input_tensor.clone()
            intermediate_activations.append(input_tensor)
            try:
                input_tensor = module(input_tensor)
                if not torch.isfinite(input_tensor).all():
                    print(f"not finite detected after {type(module)} layer {i}")
                    print()
                    break
            except Exception as e:
                print(f"Error in layer: {e}")
                break

        
        for i, module in enumerate([model.up1, model.up2, model.up3, model.up4]):
            input_tensor = input_tensor.clone()
            skip_act = intermediate_activations[-1*(i+1)]
            try:
                input_tensor = module.upconv(input_tensor)

                if not torch.isfinite(input_tensor).all():
                    print(f"not finite detected after {type(module)} layer {i} upconv")
                    print()
                    break

                diffY = skip_act.size()[2] - input_tensor.size()[2]
                diffX = skip_act.size()[3] - input_tensor.size()[3]

                input_tensor = F.pad(input_tensor, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
                input_tensor = torch.cat([skip_act, input_tensor], dim=1)

                # assume skip and skipcat are False
                for j, layer in enumerate(module.ncbr.layers):
                    #input_tensor = layer(input_tensor)
                    #input_tensor = torch.clamp(x, min=-65000, max=65000)
                    for k, sublayer in enumerate([layer.conv1, layer.bnorm1, layer.relu1]):
                        print(torch.max(torch.abs(input_tensor)))
                        print(torch.max(torch.abs(sublayer.weight.data)))

                        input_tensor = sublayer(input_tensor)
                        if not torch.isfinite(input_tensor).all():
                            print(f"{torch.numel(input_tensor)-torch.sum(torch.isfinite(input_tensor))}/{torch.numel(input_tensor)} not finite detected after {type(sublayer)} sublayer {k}")
                            if isinstance(sublayer, torch.nn.BatchNorm2d):
                                print(f"running stats: {sublayer.training}")
                            break
                    if not torch.isfinite(input_tensor).all():
                        print(f"not finite detected in {type(layer)} layer {j}")
                        break

                if not torch.isfinite(input_tensor).all():
                    print(f"not finite detected in {type(module)} layer {i} ncbr ({module.ncbr.skip}, {module.ncbr.skipcat})")
                    break
            except Exception as e:
                print(f"Error in {type(module)} layer{i}: {e}")
                break

def load_im(im_path, do_clahe=False):
    """
    Load an image from a file.

    Args:
    im_path: str, path to the image
    do_clahe: bool, whether to apply CLAHE

    Returns:
    im: np.ndarray, image
    """
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    assert im.dtype == np.uint8 or im.dtype == np.uint16
    if do_clahe:
        return clahe(im)
    else:
        return im

@timing
def write_im(im_path, im):
    """
    Write an image to a file.

    Args:
    im_path: str, path to the image
    im: np.ndarray, image
    """
    cv2.imwrite(im_path, im)


def resize_im(im, shape):
    """Resize an image to a specific shape.

    Args:
        im (np.ndarray): image
        shape (tuple): desired shape

    Returns:
        np.ndarray: resized image
    """
    if im.shape[0] > shape[0]:
        im = im[: shape[0], :]
    elif im.shape[0] < shape[0]:
        im = np.pad(im, ((0, shape[0] - im.shape[0]), (0, 0)), mode="reflect")

    if im.shape[1] > shape[1]:
        im = im[:, : shape[1]]
    elif im.shape[1] < shape[1]:
        im = np.pad(im, ((0, 0), (0, shape[1] - im.shape[1])), mode="reflect")

    return im
