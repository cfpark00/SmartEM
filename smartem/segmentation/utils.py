
from skimage import io, morphology, color
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology  import remove_small_objects

from scipy import ndimage as ndi
from skimage.color import label2rgb
import glob

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

def watershed(img,starting_point = 0):

    mb32 = img.astype(np.int32)
    # Set the threshold for minima
    minimaThreh = 15

    # Perform reconstruction

    seed = 255 - mb32 - minimaThreh
    H = 255-morphology.reconstruction(seed, 255 - mb32)
    regional_minima = ndi.minimum_filter(H, size=3)
    mask = remove_small_objects(H == regional_minima, min_size=30)
    markers_zero = label(mask) #+ starting_point
    nlabel = 5000
    w = watershed(H, markers_zero) + starting_point
    
    # Create a mask based on a threshold
    remove = img > 200
    # Set masked region in watershed result to 0
    w[remove] = 0
    
    return w 

# def voi(labels1, labels2):
#     n = len(labels1)

#     unq1 = np.unique(labels1)
#     assert sorted(unq1) == list(range(0, len(unq1)))
#     unq2 = np.unique(labels2)
#     assert sorted(unq2) == list(range(0, len(unq2)))

#     R = np.zeros((len(unq1), len(unq2)))

#     for i, lbl1 in enumerate(tqdm(labels1, leave=False)):
#         lbl2 = labels2[i]
#         R[lbl1, lbl2] += 1

#     R /= n
#     p = np.sum(R, axis=1)
#     q = np.sum(R, axis=0)

#     vi = 0
#     for i in tqdm(range(R.shape[0]), leave=False):
#         for j in range(R.shape[1]):
#             if R[i,j] > 0:
#                 vi -= R[i,j]*(np.log(R[i,j]/p[i])+np.log(R[i,j]/q[j]))

#     return vi

def visualize(**images):
    title_to_image = {}
    title_to_label = {}

    for name, image in images.items():
        title = "_".join(name.split('_')[:-1]).replace("_", " ").title()
        if "image" in name:
            title_to_image[title] = np.squeeze(image)
        elif "label" in name:
            title_to_label[title] = np.squeeze(image)

    for key in title_to_label.keys():
        assert key in title_to_image.keys()

    f, axs = plt.subplots(nrows = len(title_to_image.keys()))
    for idx, (title, image) in enumerate(title_to_image.items()):
        axs[idx].imshow(image, cmap="gray")
        axs[idx].title.set_text(title)
        if title in title_to_label.keys():
            axs[idx].imshow(title_to_label[title], cmap="jey")



