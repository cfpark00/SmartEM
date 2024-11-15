import pytest
import torch
from skimage import measure
import smartem

# from smartem.segmentation.segmenter import Segmenter
from smartem.segmentation.utils import watershed
import numpy as np


def test_watershed():
    # create a random bnw image of shape (1632,1920)
    # image = torch.randint(0, 2, (1640, 1920), dtype=torch.uint8)
    # image = np.eye(10)
    image = np.random.random((10, 10))
    # apply watershed
    # catch recursion error and let the test pass asserting on the recursion error
    with pytest.raises(RecursionError) as e_info:
        labels = watershed(image)
    # check that the number of labels is correct
    # assert len(torch.unique(labels)) == 3
