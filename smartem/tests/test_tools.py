import pytest
import numpy as np
import torch
import cv2
import os, sys

from smartem.tools import float_to_int, int_to_float, load_im, write_im, get_prob

from smartem.offline.train_mb_error_detector.NNtools import UNet


# Test function encompassing all unit tests
def test_image_processing_functions():
    # Test float_to_int
    float_image = np.array([[0.5, 1.0], [0.0, 0.25]], dtype=np.float32)
    int_image = float_to_int(float_image)
    assert np.array_equal(int_image, np.array([[127, 255], [0, 63]], dtype=np.uint8)), "float_to_int conversion failed"

    # Test int_to_float
    int_image = np.array([[100, 200], [0, 255]], dtype=np.uint8)
    float_image = int_to_float(int_image)
    assert np.allclose(float_image, np.array([[100/255, 200/255], [0/255, 1.0]], dtype=np.float32)), "int_to_float conversion failed"

    
    # Test load_im
    test_img = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
    cv2.imwrite('temp_img.png', test_img)
    loaded_image = load_im('temp_img.png', do_clahe=False)
    assert np.array_equal(loaded_image, test_img), "load_im function failed"

    # Test write_im
    write_im('temp_img_out.png', loaded_image)
    written_image = cv2.imread('temp_img_out.png', cv2.IMREAD_UNCHANGED)
    assert np.array_equal(written_image, loaded_image), "write_im function failed"


    # Test get_prob
    temp_em2mb_net = UNet.UNet(1, 2)
    prob = get_prob(test_img, temp_em2mb_net, return_dtype=np.float32)
    # Check if all values are within the range [0, 1]
    assert np.all(prob >= 0) and np.all(prob <= 1), "Probabilities are not within the range [0, 1]"
    assert prob.shape == test_img.shape, "Output probability map shape does not match input image shape"


    if os.path.exists('temp_img.png'):
        os.remove('temp_img.png')
    if os.path.exists('temp_img_out.png'):
        os.remove('temp_img_out.png')
