import abc

#### import packages first to avoid overhead
import torch
import skimage.morphology as skmorph
import numpy as np
import warnings
import cv2
import os
import re
import logging
import dis

import importlib
from smartem.segmentation.utils import watershed


class Segmenter:

    def __init__(self, model_path=None, segmenter_function=None, device="auto"):
        self.model_path = model_path
        self.model = None
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # if "watershed" in segmenter_function.__name__.lower():
        #     print("Using watershed function...")
        # else:
        #     print("Not using custom watershed function - will invert images before segmenting...")
        self.segmenter_function = segmenter_function

        self.labels = None

    def set_model(self, model_class):
        self.model = model_class.to(self.device)
        weights = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.eval()

    def preprocess(self, img):
        if img.ndim == 2:
            if img.shape[0] % 32 != 0:
                img = img[: -(img.shape[0] % 32), :]
            if img.shape[1] % 32 != 0:
                img = img[:, : -(img.shape[1] % 32)]
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            min_axis = np.argmin(img.shape)
            if min_axis == len(img.shape) - 1:
                img = img.transpose((2, 0, 1))
            if img.shape[1] % 32 != 0:
                img = img[:, : -(img.shape[1] % 32), :]
            if img.shape[2] % 32 != 0:
                img = img[:, :, : -(img.shape[2] % 32)]
        else:
            raise ValueError("Image shape not understood")

        if (img > 1).any():
            img = img / 255.0

        return img

    def get_membranes(self, img, get_probs=False):
        """
        Processes an image to generate a binary membrane mask using a pre-trained model.

        Parameters:
        img (numpy.ndarray): Input image array. It can be of shape (H, W, C) for a color image
                            or (H, W) for a grayscale image, where H is height, W is width, and C is the number of channels.
        get_probs (bool): If True, the function also returns the output probabilities from the model. Default is False.

        Returns:
        numpy.ndarray: Binary membrane mask of shape (H, W) with values 0 or 255.
        tuple: If get_probs is True, returns a tuple containing:
            - numpy.ndarray: Binary membrane mask of shape (H, W) with values 0 or 255.
            - torch.Tensor: Output probabilities from the model of shape (1, C, H, W), where C is the number of classes.
        """
        img = self.preprocess(img)
        img = torch.as_tensor(img.copy()).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(img).cpu()
            # binarize the output based on the threshold of 0.5
            if (output.sum(dim=1) == 1).all():
                mask = output > 0.5
            else:
                output = torch.softmax(output, dim=1)
                mask = output > 0.5

        mask = mask.squeeze().numpy()[1]
        mask = mask.astype(np.uint8) * 255

        if not get_probs:
            return mask
        else:
            return mask, output

    def get_labels(self, img):
        membranes = self.get_membranes(img)

        if "watershed" not in self.segmenter_function.__name__.lower():
            membranes = 255 - membranes

        labels = self.segmenter_function(membranes)
        self.labels = labels
        return labels
