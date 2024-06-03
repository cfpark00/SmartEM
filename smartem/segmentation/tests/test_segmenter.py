import pytest
import torch
from skimage import measure
import smartem
from smartem.segmentation.segmenter import Segmenter
from unittest.mock import MagicMock
from smartem.offline.train_mb_error_detector.NNtools import UNet
import numpy as np


def test_default_initialization():
    segmenter = Segmenter()
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert str(segmenter.device) == expected_device
    assert segmenter.model is None
    assert segmenter.model_path is None
    assert segmenter.segmenter_function is None
    assert segmenter.labels is None


def test_custom_initialization():
    model_path = (
        "/home/ssawmya-local/FM_work/SmartEM/smartem/segmentation/unet_50_2.81e-02.pth"
    )
    segmenter_function = measure.label
    segmenter = Segmenter(model_path=model_path, segmenter_function=segmenter_function)
    assert segmenter.model_path == model_path
    assert segmenter.segmenter_function == segmenter_function


def test_model_loading():
    mock_model_class = MagicMock()
    mock_model_class.to.return_value = mock_model_class
    model_path = (
        "/home/ssawmya-local/FM_work/SmartEM/smartem/segmentation/unet_50_2.81e-02.pth"
    )
    segmenter = Segmenter(model_path=model_path)


@pytest.fixture
def model_files(tmp_path):
    # Setup: Create U-Net models and save their state dicts to temporary files
    em2mb_net = UNet.UNet(1, 2)
    error_net = UNet.UNet(1, 2)

    # Using tempfile.TemporaryDirectory to handle cleanup automatically
    em2mb_path = tmp_path / "em2mb_net.pth"
    error_path = tmp_path / "error_net.pth"

    torch.save(em2mb_net.state_dict(), em2mb_path)
    torch.save(error_net.state_dict(), error_path)

    # Yield paths for use in tests
    yield str(em2mb_path), str(error_path)


def test_segmentation_output(model_files):
    em2mb_path, _ = model_files
    segmenter = Segmenter(model_path=em2mb_path)
    segmenter.set_model(UNet.UNet(1, 2))
    # Create random test image going from size 256x256 to 2048x2048 by going double each time
    test_images = []
    for i in range(7, 12):
        test_image = np.random.rand(2**i, 2**i)
        test_image = (test_image * 255).astype(np.uint8)
        # Get the output probabilities
        mask, probs = segmenter.get_membranes(test_image, get_probs=True)

        # Check if probabilities are between 0 and 1
        assert (probs >= 0).all() and (
            probs <= 1
        ).all(), f"Probabilities are not between 0 and 1 for image size {2**i}x{2**i}"
        probs_sum = probs.sum(axis=1)

        # if probs_sum is not all ones, print the image and the probabilities
        if not np.allclose(probs_sum, 1, atol=1e-6):
            print(
                f"Probabilities do not sum to 1 across the channel dimension for image size {2**i}x{2**i}"
            )
            print(test_image)
            print(probs[0])
        # Check for NaNs or Infs
        assert not torch.isnan(probs).any(), "Probabilities contain NaN values"
        assert not torch.isinf(probs).any(), "Probabilities contain Inf values"

        # Check for numerical stability
        assert torch.all(probs > 1e-8), "Probabilities contain very small values"
        assert torch.all(probs < 1 - 1e-8), "Probabilities contain very large values"
