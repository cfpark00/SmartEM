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
def get_prob(image, net, return_dtype=np.uint8):
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
        mask_logits = net(
            image_torch.to(device=next(net.parameters()).device, dtype=torch.float32)
        )
        prob = (
            torch.exp(get_logprob(mask_logits))[0, 1].cpu().detach().numpy()
        )  # 1st channel for membrane
    if return_dtype == np.uint8:
        return float_to_int(prob, dtype=return_dtype)
    else:
        return prob.astype(return_dtype)


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
        im = np.pad(im, ((0, shape[0] - im.shape[0]), (0, 0)), mode="edge")

    if im.shape[1] > shape[1]:
        im = im[:, : shape[1]]
    elif im.shape[1] < shape[1]:
        im = np.pad(im, ((0, 0), (0, shape[1] - im.shape[1])), mode="edge")

    return im
