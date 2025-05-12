from smartem.online.microscope import FakeDataMicroscope
import pytest
from skimage import io
import numpy as np


@pytest.fixture
def make_dt_to_path(tmp_path):
    dt_to_path = {}
    dt_to_im = {}

    for dt in [50, 800]:
        im_path = tmp_path / f"{dt}.tif"
        im = np.random.randint(0, 255, size=(1024, 1024), dtype=np.uint8)
        io.imsave(im_path, im)
        dt_to_path[dt] = str(im_path)
        dt_to_im[dt] = im

    return dt_to_path, dt_to_im


def test_FakeDataMicroscope(tmp_path, make_dt_to_path):
    dt_to_path, dt_to_im = make_dt_to_path

    # test initialization
    FakeDataMicroscope()

    path_nonexistant_im = tmp_path / "ne.tif"
    with pytest.raises(AssertionError):  # file does not yet exist
        FakeDataMicroscope({"images_ns": {800: str(path_nonexistant_im)}})

    # test get_image
    microscope = FakeDataMicroscope({"images_ns": dt_to_path})
    for dt in dt_to_path.keys():
        im = microscope.get_image({"dwell_time": dt * 1e-9})
        assert (im == dt_to_im[dt]).all()
    with pytest.raises(ValueError):  # dwell time does not exist
        microscope.get_image({"dwell_time": 1000e-9})


def test_FakeDataMicroscope_sleep(make_dt_to_path):
    dt_to_path, dt_to_im = make_dt_to_path

    # test get_image
    microscope = FakeDataMicroscope({"images_ns": dt_to_path}, sleep=True)
    for dt in dt_to_path.keys():
        im = microscope.get_image({"dwell_time": dt * 1e-9, "resolution": (2048, 1768)})
        assert (im == dt_to_im[dt]).all()
    with pytest.raises(ValueError):  # dwell time does not exist
        microscope.get_image({"dwell_time": 1000e-9})


def test_FakeDataMicroscope_resize(make_dt_to_path):
    dt_to_path, _ = make_dt_to_path

    # test get_image
    microscope = FakeDataMicroscope({"images_ns": dt_to_path}, pad_images=True)
    for shp in [(512, 256), (4096, 2048)]:
        im = microscope.get_image({"dwell_time": 50 * 1e-9, "resolution": shp})
        assert im.shape == shp
