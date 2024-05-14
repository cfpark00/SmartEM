from skimage import io, morphology, color
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects

from scipy import ndimage as ndi
from skimage.color import label2rgb
import glob

import numpy as np
from tqdm import tqdm
import random

import matplotlib.pyplot as plt


def watershed(img, starting_point=0):

    mb32 = img.astype(np.int32)
    # Set the threshold for minima
    minimaThreh = 15

    # Perform reconstruction

    seed = 255 - mb32 - minimaThreh
    H = 255 - morphology.reconstruction(seed, 255 - mb32)
    regional_minima = ndi.minimum_filter(H, size=3)
    mask = remove_small_objects(H == regional_minima, min_size=30)
    markers_zero = label(mask)  # + starting_point
    nlabel = 5000
    w = watershed(H, markers_zero) + starting_point

    # Create a mask based on a threshold
    remove = img > 200
    # Set masked region in watershed result to 0
    w[remove] = 0

    return w


def visualize(**images):
    # Inputs should be named <title>_<'label' or 'image' or 'pred'> so this function can match labels to images
    title_to_image = {}
    title_to_label = {}
    title_to_mb = {}
    title_to_err = {}

    for name, image in images.items():
        title = "_".join(name.split("_")[:-1]).replace("_", " ").title()
        if "image" in name:
            title_to_image[title] = np.squeeze(image)
        elif "label" in name:
            title_to_label[title] = np.squeeze(image)
        elif "pred" in name:
            title_to_mb[title] = np.squeeze(image)
        elif "error" in name:
            title_to_err[title] = (np.squeeze(image[0]), np.squeeze(image[1]))

    for key in title_to_label.keys():
        assert key in title_to_image.keys()
    for key in title_to_mb.keys():
        assert key in title_to_image.keys()
    for key in title_to_err.keys():
        assert key in title_to_image.keys()

    f, axs = plt.subplots(nrows=2, ncols=len(title_to_image.keys()))
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs, axis=1)

    for idx, (title, image) in enumerate(title_to_image.items()):
        axs[0, idx].imshow(image, cmap="gray")
        axs[0, idx].title.set_text(title)

        if title in title_to_label.keys():
            axs[0, idx].imshow(title_to_label[title], cmap="jet", alpha=0.5)
        if title in title_to_mb.keys():
            axs[1, idx].imshow(title_to_mb[title], cmap="gray")
            axs[1, idx].title.set_text("MB Predictions w/get_error_GT (merge/yellow, split/red)")
        if title in title_to_err.keys():
            axs[1, idx].imshow(title_to_err[title][0], cmap="Wistia", alpha=0.5)
            axs[1, idx].imshow(title_to_err[title][1], cmap="autumn", alpha=0.5)
        

        xax = axs[0, idx].axes.get_xaxis()
        xax = xax.set_visible(False)
        yax = axs[0, idx].axes.get_yaxis()
        yax = yax.set_visible(False)
        xax = axs[1, idx].axes.get_xaxis()
        xax = xax.set_visible(False)
        yax = axs[1, idx].axes.get_yaxis()
        yax = yax.set_visible(False)

    f.set_figheight(10)
    f.set_figwidth(5 * len(title_to_image.keys()))
    f.tight_layout()


def shuffle_labels(im):
    im_shuffled = np.zeros(im.shape)

    for lbl in np.unique(im):
        if lbl > 0:
            im_shuffled[im == lbl] = random.randint(1, 255)

    return im_shuffled

# def get_ious()
