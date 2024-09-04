import pytest
import numpy as np
import os
import json
from pathlib import Path
import time

from smartem.smartem_par import SmartEMPar, par_test
from smartem.online import microscope as microscope_client
from smartem.online import get_rescan_maps

from test_smartem import get_default_params, get_smartem

repo_dir = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]


@pytest.fixture
def get_smartem_par():
    # initializing fake random microscope with sleep on
    params = {"W": 1024, "H": 1024, "dtype": np.uint16, "sleep": True}
    microscope = microscope_client.FakeRandomMicroscope(params=params)

    # initializing get_rescan_map
    params = {"type": "half", "fraction": 0.5}
    get_rescan_map = get_rescan_maps.GetRescanMapTest(params=params)
    smart_em = SmartEMPar(microscope, get_rescan_map)

    assert smart_em.microscope == microscope
    assert smart_em.get_rescan_map == get_rescan_map

    yield smart_em


def test_par_test():
    partest = par_test()
    locs = [i for i in range(10)]
    sleep_a, sleep_b = 1, 0.5
    total_time_serial = len(locs) * (sleep_a + sleep_b)

    tic = time.time()
    rescan_masks = partest.run(locs, sleep_a, sleep_b)
    toc = time.time()
    diff = toc - tic

    assert all([i + 1 == j for i, j in zip(locs, rescan_masks)])
    assert diff < total_time_serial


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
    # simulate the time it takes to generate a rescan map, in seconds
    sleep_time = 0.25  # this choice is large enough to be noticeable, but not too large that the test takes too long

    smart_em = get_smartem
    smart_em.initialize()
    smart_em_par = get_smartem_par
    smart_em_par.initialize()

    params = get_default_params

    # Don't acquire as many tiles as there will be sleeping involved
    with open(
        repo_dir / "examples/default_imaging_params_short.json",
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
