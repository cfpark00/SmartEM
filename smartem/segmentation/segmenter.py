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
from utils import watershed



class Segmenter:

    def __init__(self, model_path = None, segmenter_function = None):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.segmenter_function = segmenter_function
        self.labels = None

    def set_model(self, model_class):
        # self.model_path = model_path
        self.model = model_class.to(self.device)
        weights = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.eval()

    def preprocess(self,img):
        if img.ndim == 2:
            if img.shape[0] % 32 != 0:
                img = img[:-(img.shape[0] % 32), :]
            if img.shape[1] % 32 != 0:
                img = img[:, :-(img.shape[1] % 32)]
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            min_axis = np.argmin(img.shape)
            if min_axis == len(img.shape) - 1:
                img = img.transpose((2, 0, 1))
            if img.shape[1] % 32 != 0:
                img = img[:, :-(img.shape[1] % 32), :]
            if img.shape[2] % 32 != 0:
                img = img[:, :, :-(img.shape[2] % 32)]
        else:
            raise ValueError("Image shape not understood")

        if (img > 1).any():
            img = img / 255.0

        return img

    def get_membranes(self, img):
        # print(img.shape)
        img = self.preprocess(img)
        img = torch.as_tensor(img.copy()).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(img).cpu()
            if (output >= 0).all() and (output <= 1).all():
                mask = output
            else:
                mask = torch.sigmoid(output) > 0.5

        mask = mask.squeeze().numpy()[1]
        mask = mask.astype(np.uint8) * 255


        return mask

    def get_labels(self, img):
        membranes = self.get_membranes(img)
        # print the type of 
        # if self.segmenter_function.__code__.co_code == watershed.__code__.co_code:
        if "watershed" in self.segmenter_function.__name__.lower():
            print("Using watershed function")
        else:
            print("Inverting the image as not using custom watershed function")
            membranes = 255 - membranes
            
        
        labels = self.segmenter_function(membranes)
        self.labels = labels
        return labels
    
    def calculate_voi(self,gt_labels):
        if self.labels is None:
            raise ValueError("No labels to compare")
        return voi(gt_labels, self.labels)


        
