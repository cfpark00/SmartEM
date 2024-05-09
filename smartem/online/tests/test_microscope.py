from smartem.online.microscope import FakeDataMicroscope
import pytest
from skimage import io
import numpy as np


def test_FakeDataMicroscope_init(tmp_path):
    im_shape = (1024, 1024)
    dt_to_path = {50: None, 800: None}

    # test initialization
    FakeDataMicroscope()

    path_nonexistant_im = tmp_path / "ne.tif"
    with pytest.raises(AssertionError):  # file does not yet exist
        FakeDataMicroscope({"images_ns": {800: str(path_nonexistant_im)}})

    # save fake data
    dt_to_im = {key: val for key, val in dt_to_path.items()}
    for dt in dt_to_path.keys():
        path = str(tmp_path / f"{dt}.tif")
        im = np.random.randint(0, 255, size=im_shape, dtype=np.uint8)
        io.imsave(path, im)

        dt_to_path[dt] = path
        dt_to_im[dt] = im

    # test get_image
    microscope = FakeDataMicroscope({"images_ns": dt_to_path})
    for dt in dt_to_path.keys():
        im = microscope.get_image({"dwell_time": dt * 1e-9})
        assert (im == dt_to_im[dt]).all()
    with pytest.raises(ValueError):  # dwell time does not exist
        microscope.get_image({"dwell_time": 1000e-9})
