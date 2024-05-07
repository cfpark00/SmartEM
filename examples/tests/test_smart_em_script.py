import pytest
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "examples")))
# from ..examples.smart_em_script import get_microscope


from smartem.online.microscope import (
    FakeRandomMicroscope,
    FakeDataMicroscope,
    ThermoFisherVerios,
)


def test_get_microscope():
    # test fake microscope
    microscope = get_microscope("fake")
    assert isinstance(
        microscope, FakeRandomMicroscope
    ), "Returned object is not a FakeRandomMicroscope"
    expected_params = {"W": 1024, "H": 1024, "dtype": np.uint16}
    assert (
        microscope.params == expected_params
    ), "Microscope parameters do not match expected for 'fake' type"

    # test fake data microscope
    # save current path for switching later
    current_path = os.getcwd()
    # make the current path the parent's parent path
    os.chdir(os.path.abspath(os.path.join("..")))
    microscope = get_microscope("fakedata")
    assert isinstance(
        microscope, FakeDataMicroscope
    ), "Returned object is not a FakeDataMicroscope"

    # switch to the current path
    os.chdir(current_path)
