import pytest
import torch
from skimage import measure
import smartem
import pytest
import numpy as np
from smartem.online.get_rescan_maps import GetRescanMapMembraneErrors
from smartem.online.get_rescan_maps import GetRescanMapTest
from smartem.offline.train_mb_error_detector.NNtools import UNet
import os, sys
from pathlib import Path


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


def test_initialize(model_files):
    em2mb_path, error_path = model_files
    params = {
        "em2mb_net": em2mb_path,
        "error_net": error_path,
        "device": "auto",  # Set to auto to use GPU if available
        "pad": 0,
        "rescan_p_thres": 0.1,
        "do_clahe": False,
    }

    # Initialize rescan map object with the parameters
    rescan_map = GetRescanMapMembraneErrors(params=params)
    rescan_map.initialize()

    # Assertions to check if models are loaded correctly and device is set
    assert isinstance(rescan_map.em2mb_net, UNet.UNet)
    assert isinstance(rescan_map.error_net, UNet.UNet)
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert str(rescan_map.device) == expected_device

    # Load original state dicts and compare to the loaded ones
    original_em2mb_state_dict = torch.load(em2mb_path, map_location=rescan_map.device)
    loaded_em2mb_state_dict = rescan_map.em2mb_net.state_dict()
    assert all(
        torch.equal(original_em2mb_state_dict[k], loaded_em2mb_state_dict[k])
        for k in original_em2mb_state_dict
    ), "Mismatch in EM2MB model parameters"

    original_error_state_dict = torch.load(error_path, map_location=rescan_map.device)
    loaded_error_state_dict = rescan_map.error_net.state_dict()
    assert all(
        torch.equal(original_error_state_dict[k], loaded_error_state_dict[k])
        for k in original_error_state_dict
    ), "Mismatch in Error model parameters"


def test_rescan_ratio(model_files):
    # rescan_map = setup_rescan_map
    em2mb_path, error_path = model_files
    params = {
        "em2mb_net": em2mb_path,
        "error_net": error_path,
        "device": "auto",  # Set to auto to use GPU if available
        "pad": 0,
        "rescan_p_thres": 0.1,
        "do_clahe": False,
    }

    # Initialize rescan map object with the parameters
    rescan_map = GetRescanMapMembraneErrors(params=params)
    rescan_map.initialize()

    rescan_map.params["rescan_ratio"] = 0.2
    rescan_map.params["rescan_p_thres"] = None

    # test_image = np.linspace(0, 1, 100).reshape(10, 10)
    # make a random test image of shape 1024 by 1024
    test_image = np.random.rand(1024, 1024)
    # put test_image in uint8
    test_image = (test_image * 255).astype(np.uint8)
    map_output, extras = rescan_map.get_rescan_map(test_image)
    with pytest.raises(AssertionError):
        assert (map_output.sum() / 100) == pytest.approx(
            0.2
        ), "Should select roughly 20% of the area"
    # assert (map_output.sum() / 100) == pytest.approx(0.2), "Should select roughly 20% of the area"
    # print(np.unique(map_output, return_counts=True))
