import torch
import glob
import os
import numpy as np
from PIL import Image
import scipy.ndimage as sim
from tqdm import tqdm

import numpy as np


class MustExcludeDataset(torch.utils.data.Dataset):
    def __init__(self, datadict):
        super().__init__()

        self.ims = []
        self.masks = []

        self.ems_masks = datadict
        for key, patch_dict in self.ems_masks.items():
            for patch_number, patch_em_label in self.ems_masks[key].items():
                self.ims.append(patch_em_label["image"])
                self.masks.append(patch_em_label["mask"])
        print("Finding unique values")
        unique_values = np.unique([np.unique(x) for x in tqdm(self.masks)])

        assert len(self.ims) == len(self.masks)
        self.n_samples = len(self.ims)
        self.mask_values = list(sorted(unique_values))
        print("Unique values", self.mask_values)

    @staticmethod
    def preprocess(mask_values, img, is_mask):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # img = np.asarray(pil_img)
        w, h = img.shape
        if is_mask:
            mask = np.zeros((h, w), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        if (not isinstance(i, int)) or i < 0 or i >= self.n_samples:
            raise IndexError

        image = self.ims[i]
        mask = self.masks[i]

        image = self.preprocess(self.mask_values, image, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, is_mask=True)

        #         image = torch.tensor(self.ims[i]).float().contiguous()
        #         mask = torch.tensor(self.masks[i]).long().contiguous()
        return {
            "image": torch.as_tensor(image.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }
