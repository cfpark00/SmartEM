import pytest
import torch
from skimage import measure
import smartem
from smartem.segmentation.segmenter import Segmenter
from unittest.mock import MagicMock

def test_default_initialization():
    segmenter = Segmenter()
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert str(segmenter.device) == expected_device
    assert segmenter.model is None
    assert segmenter.model_path is None
    assert segmenter.segmenter_function is None
    assert segmenter.labels is None

def test_custom_initialization():
    model_path = "/home/ssawmya-local/FM_work/SmartEM/smartem/segmentation/unet_50_2.81e-02.pth"
    segmenter_function = measure.label
    segmenter = Segmenter(model_path=model_path, segmenter_function=segmenter_function)
    assert segmenter.model_path == model_path
    assert segmenter.segmenter_function == segmenter_function

def test_model_loading():
    mock_model_class = MagicMock()
    mock_model_class.to.return_value = mock_model_class
    model_path = "/home/ssawmya-local/FM_work/SmartEM/smartem/segmentation/unet_50_2.81e-02.pth"
    segmenter = Segmenter(model_path=model_path)
    # segmenter.set_model(mock_model_class)
    # mock_model_class.to.assert_called_with(segmenter.device)
    # mock_model_class.load_state_dict.assert_called_once()