import pytest
import numpy as np
import os
import json
from pathlib import Path

from smartem.smartem import SmartEM
from smartem.online import microscope as microscope_client
from smartem.online import get_rescan_maps

repo_dir = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]


@pytest.fixture
def get_smartem():
    # initializing fake random microscope with sleep on
    params = {"W": 1024, "H": 1024, "dtype": np.uint16, "sleep": True}
    microscope = microscope_client.FakeRandomMicroscope(params=params)

    # initializing get_rescan_map
    params = {"type": "half", "fraction": 0.5}
    get_rescan_map = get_rescan_maps.GetRescanMapTest(params=params)
    smart_em = SmartEM(microscope, get_rescan_map)

    assert smart_em.microscope == microscope
    assert smart_em.get_rescan_map == get_rescan_map

    yield smart_em


@pytest.fixture
def get_default_params():
    with open(
        repo_dir / "examples/default_smartem_params.json",
        "r",
    ) as f:
        params = json.load(f)
        if "resolution" in params:
            params["resolution"] = tuple(params["resolution"])
        params["plot"] = False

    with open(
        repo_dir / "examples/default_imaging_params.json",
        "r",
    ) as f:
        params_imaging = json.load(f)
        params.update(params_imaging)

    return params


def test_smart_em_operations_using_fake_data_and_microscope(get_smartem):
    smart_em = get_smartem
    smart_em.initialize()

    # test prepare_acquisition
    smart_em.prepare_acquisition()

    # testing acquire function
    params = {"fast_dwt": 50e-9, "slow_dwt": 800e-9, "plot": True, "verbose": 1}
    fast_em, rescan_em, rescan_map, additional = smart_em.acquire(params=params)

    # test whether fast_em is a numpy array of size 1024x1024 of type np.uint16
    assert isinstance(fast_em, np.ndarray), "fast_em is not a numpy array"
    assert fast_em.shape == (1024, 1024), "fast_em is not of size 1024x1024"
    assert fast_em.dtype == np.uint16, "fast_em is not of type np.uint16"

    # test whether rescan_em is a numpy array of size 1024x1024
    assert isinstance(rescan_em, np.ndarray), "rescan_em is not a numpy array"
    assert rescan_em.shape == (1024, 1024), "rescan_em is not of size 1024x1024"
    assert rescan_em.dtype == np.uint16, "rescan_em is not of type np.uint16"

    # test whether rescan_map is a numpy array of size 1024x1024 of type bool
    assert isinstance(rescan_map, np.ndarray), "rescan_map is not a numpy array"
    assert rescan_map.shape == (1024, 1024), "rescan_map is not of size 1024x1024"
    assert (
        rescan_map.dtype == bool
    ), f"rescan_map is not of type bool, rather of type {rescan_map.dtype} "

    # test show_smart function from the return additional dictionary
    fig = additional["fig"]
    # Check the number of subplots
    assert len(fig.axes) == 5, "The number of subplots is incorrect"

    # Expected titles
    expected_titles = [
        f"fast_em, dwell_time = {params['fast_dwt']*1e9:.0f} ns",
        "rescan_map",
        f"slow_em, dwell_time = {params['slow_dwt']*1e9:.0f} ns",
        "merged_em",
        "merged_em - fast_em",
    ]

    # Expected titles and data
    expected_data = [
        (fast_em, f"fast_em, dwell_time = {params['fast_dwt']*1e9:.0f} ns"),
        (rescan_map, "rescan_map"),
        (rescan_em, f"slow_em, dwell_time = {params['slow_dwt']*1e9:.0f} ns"),
        (None, "merged_em"),  # Merged data is processed within the function
        (
            None,
            "merged_em - fast_em",
        ),  # Difference data is processed within the function
    ]

    # Check the titles and contents of each subplot
    for ax, (expected_array, expected_title) in zip(fig.axes, expected_data):
        assert (
            ax.get_title() == expected_title
        ), f"Expected title '{expected_title}' but got '{ax.get_title()}'"
        displayed_image = ax.get_images()[0].get_array().data
        if expected_array is not None:
            np.testing.assert_array_almost_equal(
                displayed_image,
                expected_array,
                decimal=5,
                err_msg=f"Data mismatch in plot titled '{expected_title}'",
            )


def test_smart_em_acquire_many_grids(get_smartem, get_default_params, tmp_path):
    smart_em = get_smartem
    smart_em.initialize()

    params = get_default_params

    smart_em.acquire_many_grids(
        coordinates=params["coordinates"], params=params, save_dir=tmp_path
    )
