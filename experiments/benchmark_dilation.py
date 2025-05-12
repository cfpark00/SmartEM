import numpy as np
import matplotlib.pyplot as plt

# from scipy.ndimage import binary_dilation
# from skimage.morphology import dilation, disk
import scipy
import skimage
import cv2
import torch
import torch.nn.functional as F
import time


def generate_separated_points(image_size, num_points, min_distance):
    points = []

    while len(points) < num_points:
        # Generate a candidate point
        x, y = np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1])
        valid = True

        # Check the distance from this point to all previously selected points
        for px, py in points:
            if abs(x - px) < min_distance and abs(y - py) < min_distance:
                valid = False
                break

        # If valid, add the point to the list
        if valid:
            points.append((x, y))

    return points


def get_random_binary_image():
    image_size = np.array([2048, 1768])
    binary_image = np.zeros(image_size)

    # find ten pairs of random points
    npair = 10
    points = generate_separated_points(image_size - 125, npair, 125)
    x, y = zip(*points)
    for i in range(npair):
        binary_image[x[i] : x[i] + 123, y[i] : y[i] + 123] = 1

    # plt.imshow(binary_image, cmap='gray')

    return binary_image


def compute_white_pixel_proportion(binary_image):
    # proportion of white pixels
    white_pixels = np.sum(binary_image == 1)
    total_pixels = binary_image.size
    return white_pixels * 100 / total_pixels


n = 2
binary_image = get_random_binary_image().astype(np.uint8)
print(compute_white_pixel_proportion(binary_image))

# Structuring element (e.g., a disk with radius 1)
structuring_element = np.ones((50, 50), dtype=np.uint8)  # A disk of radius 1 pixel

# Dilate the white pixels

data_time = []
for i in range(n):
    tic = time.time()
    dilated_image1 = skimage.morphology.binary_dilation(
        binary_image, structuring_element
    )
    data_time.append(time.time() - tic)

print(f"skimage.morphology: {np.mean(data_time)},{np.std(data_time)}")

data_time = []
for i in range(n):
    tic = time.time()
    dilated_image2 = scipy.ndimage.binary_dilation(
        binary_image, structure=structuring_element
    )
    data_time.append(time.time() - tic)

print(f"scipy.ndimage.binary_dilation: {np.mean(data_time)},{np.std(data_time)}")


data_time = []
for i in range(n):
    tic = time.time()
    dilated_image3 = cv2.dilate(binary_image, structuring_element, iterations=1)
    data_time.append(time.time() - tic)

print(f"cv2.dilate: {np.mean(data_time)},{np.std(data_time)}")

# assert np.all(dilated_image1 == dilated_image2)

for im in [dilated_image2, dilated_image3.astype(bool)]:
    print(im.shape)
    print(im.dtype)
    print(np.amin(im))
    print(np.amax(im))
    print(np.sum(im))
    print(np.sum(im) / im.size)

print(np.sum(dilated_image2 == dilated_image3.astype(bool)) / dilated_image2.size)
