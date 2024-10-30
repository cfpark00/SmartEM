import pytest
import numpy as np
import sys, os
import json
from pathlib import Path
import time

# sys.path.append(os.path.abspath(os.path.join('../..', 'examples')))
# from smart_em_script import get_microscope, get_get_rescan_map
import smartem
from smartem.smartem import SmartEM
from smartem.smartem_par import SmartEMPar
from smartem.online import microscope as microscope_client
from smartem.online import get_rescan_maps
from test_smartem import get_default_params

repo_dir = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]


@pytest.fixture
def get_smartem():
    # initializing fake random microscope with sleep on
    params = {"ip": "localhost", "sleep": True}
    microscope = microscope_client.ThermoFisherVerios(params=params)

    # initializing get_rescan_map
    params = {"type": "half", "fraction": 0.5}
    get_rescan_map = get_rescan_maps.GetRescanMapTest(params=params)
    smart_em = SmartEM(microscope, get_rescan_map)

    assert smart_em.microscope == microscope
    assert smart_em.get_rescan_map == get_rescan_map

    yield smart_em


@pytest.fixture
def get_smartem_par():
    # initializing fake random microscope with sleep on
    params = {"ip": "localhost", "sleep": True}
    microscope = microscope_client.ThermoFisherVerios(params=params)

    # initializing get_rescan_map
    params = {"type": "half", "fraction": 0.5}
    get_rescan_map = get_rescan_maps.GetRescanMapTest(params=params)
    smart_em = SmartEMPar(microscope, get_rescan_map)

    assert smart_em.microscope == microscope
    assert smart_em.get_rescan_map == get_rescan_map

    yield smart_em


def test_smart_em_operations_using_fake_data_and_verios(
    get_smartem, get_default_params
):
    smart_em = get_smartem
    params = get_default_params
    params["plot"] = True
    smart_em.initialize()

    # test prepare_acquisition
    smart_em.prepare_acquisition()

    # testing acquire function
    fast_em, rescan_em, rescan_map, additional = smart_em.acquire(params=params)

    # test whether fast_em is a numpy array of size 1768x2048 of type np.uint16
    assert isinstance(fast_em, np.ndarray), "fast_em is not a numpy array"
    assert fast_em.shape == (1768, 2048), "fast_em is not of size 1768x2048"
    assert fast_em.dtype == np.uint8, "fast_em is not of type np.uint16"

    # test whether rescan_em is a numpy array of size 1768x2048
    assert isinstance(rescan_em, np.ndarray), "rescan_em is not a numpy array"
    assert rescan_em.shape == (1768, 2048), "rescan_em is not of size 1768x2048"
    assert rescan_em.dtype == np.uint8, "rescan_em is not of type np.uint16"

    # test whether rescan_map is a numpy array of size 1768x2048 of type bool
    assert isinstance(rescan_map, np.ndarray), "rescan_map is not a numpy array"
    assert rescan_map.shape == (1768, 2048), "rescan_map is not of size 1768x2048"
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


def test_smart_em_par_acquire_many_grids(get_smartem_par, get_default_params, tmp_path):
    smart_em = get_smartem_par
    smart_em.initialize()

    params = get_default_params

    smart_em.acquire_many_grids(
        coordinates=params["coordinates"], params=params, save_dir=tmp_path
    )


def test_smart_em_acquire_many_grids_time_comp(
    get_smartem, get_smartem_par, get_default_params, tmp_path
):
    sleep_time = 30

    smart_em = get_smartem
    smart_em.initialize()
    smart_em_par = get_smartem_par
    smart_em_par.initialize()

    params = get_default_params
    with open(
        repo_dir / "examples/default_imaging_params_single.json",
        "r",
    ) as f:
        params_imaging = json.load(f)
        params.update(params_imaging)

    params["sleep_time"] = sleep_time
    smart_em.get_rescan_map.params["sleep_time"] = sleep_time
    smart_em_par.get_rescan_map.params["sleep_time"] = sleep_time

    # serial first
    tic = time.time()
    smart_em.acquire_many_grids(
        coordinates=params["coordinates"], params=params, save_dir=tmp_path
    )
    toc_serial = time.time()
    smart_em_par.acquire_many_grids(
        coordinates=params["coordinates"], params=params, save_dir=tmp_path
    )
    toc_par = time.time()
    assert toc_par - toc_serial < 0.85 * (toc_serial - tic)

    # parallel first
    tic = time.time()
    smart_em_par.acquire_many_grids(
        coordinates=params["coordinates"], params=params, save_dir=tmp_path
    )
    toc_par = time.time()
    smart_em.acquire_many_grids(
        coordinates=params["coordinates"], params=params, save_dir=tmp_path
    )
    toc_serial = time.time()
    assert toc_par - tic < 0.85 * (toc_serial - toc_par)
