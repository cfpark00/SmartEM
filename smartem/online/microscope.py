import abc
import copy

### package imports ###
import numpy as np
import os
import warnings
from pathlib import Path

from smartem import tools

import copy
import time

from smartem.timing import timing, time_block


class BaseMicroscope(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def prepare_acquisition(self):
        pass

    @abc.abstractmethod
    def get_image(self, params):
        pass

    @abc.abstractmethod
    def move(self, params):
        pass

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class FakeRandomMicroscope(BaseMicroscope):
    """This class acts as a synthetic microscope, returning random image data.

    Attributes:
        params (dict): image data details including width, height, and datatype

    Methods:
        prepare_acquisition: placeholder
        get_image: returns random image data
        initialize: placeholder
        close: placeholder
    """

    default_params = {
        "W": 1024,
        "H": 1024,
        "dtype": np.uint16,
    }

    def __init__(self, params=None):
        super().__init__()
        self.params = self.default_params
        if params is not None:
            self.params.update(params)

    def prepare_acquisition(self):
        pass

    def get_image(self, params):
        W = params["W"] if "W" in params else 1024
        H = params["H"] if "H" in params else 1024
        dtype = params["dtype"] if "dtype" in params else np.uint16
        image = np.random.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, (W, H), dtype=dtype
        )
        if "rescan_map" in params.keys():
            if "sleep" in self.params.keys():
                # simulate slow scan, though 1e7 factor is somewhat arbitrary
                time.sleep(
                    params["slow_dwt"] * 1e7 * 0.05
                )  # 0.05 represents 5% of pixels
            image[~params["rescan_map"]] = np.iinfo(dtype).min
            return image
        else:
            if "sleep" in self.params.keys():
                # simulate fast scan, though 1e7 factor is somewhat arbitrary
                time.sleep(params["fast_dwt"] * 1e7)
            return image

    def move(self, **kwargs):
        pass

    def initialize(self):
        pass

    def close(self):
        pass


class FakeDataMicroscope(BaseMicroscope):
    """This class acts as a synthetic microscope, returning data from saved files.

    Attributes:
        params (dict): paths of images at various dwell times

    Methods:
        initialize: placeholder
        close: placeholder
        prepare_acquisition: placeholder
        get_image: read file of given dwell time
    """

    default_params = {"tempfile": "./tempfile.bmp", "images_ns": {}}

    def __init__(self, params=None, sleep=False, pad_images=False):
        """Initialize microscope with paths.

        Args:
            params (dict, optional): paths of images at various dwell times. Defaults to None.
            sleep (bool): Whether the microscope should sleep during function calls.
            pad_images (bool): Whether the images returned by this microscope should be padded to the requested shape.
        """
        super().__init__()
        self.params = self.default_params
        if params is not None:
            self.params.update(params)
        for key in self.params["images_ns"].keys():
            assert os.path.exists(
                self.params["images_ns"][key]
            ), f"File {self.params['images_ns'][key]} does not exist"

        self.sleep = sleep
        self.pad_images = pad_images

    @timing
    def prepare_acquisition(self):
        """Calls auto_stig, auto_focus, and auto_contrast_brightness."""
        self.auto_stig()
        self.auto_focus()
        self.auto_contrast_brightness()

    @timing
    def get_image(self, params):
        """Read file and return data for image of given dwell time.

        Args:
            params (dict): desired dwell time

        Raises:
            ValueError: if file path for given dwell time is not available
            ValueError: if file path for given dwell time does not exist

        Returns:
            np.ndarray: image
        """
        dwt = params["dwell_time"]
        dwt_ns = int(dwt * 1e9)
        if dwt_ns not in self.params["images_ns"].keys():
            raise ValueError(f"No data for dwell time {dwt_ns} ns")
        file_path = self.params["images_ns"][dwt_ns]
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")

        start = time.time()
        if "rescan_map" in params.keys():
            rescan_map = params["rescan_map"]

            rescan_map = (rescan_map.astype(np.uint8) * 255)[:, :, None].repeat(
                3, axis=2
            )
            tools.write_im(self.params["tempfile"], rescan_map)
            im = tools.load_im(file_path)
        else:
            im = tools.load_im(file_path)

        if self.pad_images:
            im = tools.resize_im(im, params["resolution"])

            if "rescan_map" in params.keys():
                im[rescan_map[:, :, 0] == 0] = 0

        if self.sleep:
            num_pixels = np.prod(params["resolution"])
            if "rescan_map" in params.keys():
                rescan_frac = np.sum(params["rescan_map"]) / num_pixels
                if num_pixels == 2048 * 1768:
                    im_time = 1.691
                elif num_pixels == 4096 * 3536:
                    im_time = 4.69
                else:
                    raise ValueError(
                        f"Sleep not supported for resolution {params['resolution']}"
                    )
            else:
                rescan_frac = 1
                if num_pixels == 2048 * 1768:
                    im_time = 0.631
                elif num_pixels == 4096 * 3536:
                    im_time = 1.24
                else:
                    raise ValueError(
                        f"Sleep not supported for resolution {params['resolution']}"
                    )
            im_time += dwt * num_pixels * rescan_frac
            elapsed = (
                time.time() - start
            )  # remove the image loading time from the sleep time
            if elapsed < im_time:
                time.sleep(im_time - elapsed)

        return im

    @timing
    def move(self, x, y, z=None, r=None, t=None):
        if self.sleep:
            time.sleep(0)

    @timing
    def auto_focus(self):
        if self.sleep:
            time.sleep(0)

    @timing
    def auto_contrast_brightness(self):
        if self.sleep:
            time.sleep(0)

    @timing
    def auto_stig(self):
        if self.sleep:
            time.sleep(0)

    @timing
    def initialize(self):
        pass

    def close(self):
        pass


class ThermoFisherVerios(BaseMicroscope):
    """Communicates with Thermo Fisher Verios microscope to implement SmartEM pipeline.

    Attributes:
        params (dict): microscope and SmartEM parameters

    Methods:
        initialize: connect to EM
        close: disconnect from EM
        prepare_acquisition: prepare EM for imaging
        auto_focus: execute autofocus
        auto_contrast_brightness: adjust contrast
        auto_stig: run stig
        get_image: read file of given dwell time
    """

    default_params = {
        "tempfile": "./tempfile.bmp",
        "AS_final_horizontal_field_width_stig": 5.0e-6,
        "AF_final_horizontal_field_width_focus": 8.0e-6,
        "slow_dwt_focus_ns": 1000,
        "fast_dwt_contrast_ns": 100,
    }

    def __init__(self, params=None):
        super().__init__()
        self.params = self.default_params
        if params is not None:
            self.params.update(params)
        assert "ip" in self.params.keys(), "ip address of microscope must be provided"
        assert os.path.exists(
            os.path.split(self.params["tempfile"])[0]
        ), "tempfile folder does not exist"

        import autoscript_sdb_microscope_client as sdb_client
        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        import autoscript_sdb_microscope_client.enumerations as sdb_enums
        import autoscript_sdb_microscope_client.sdb_microscope_client as sdb_microscope_client
        from autoscript_sdb_microscope_client.enumerations import ImagingDevice
        import autoscript_sdb_microscope_client.structures as sdb_structures
        from autoscript_sdb_microscope_client.structures import GrabFrameSettings

        self.sdb_client = sdb_client
        self.SdbMicroscopeClient = SdbMicroscopeClient
        self.sdb_enums = sdb_enums
        self.BitmapPatternDefinition = sdb_microscope_client.BitmapPatternDefinition
        self.ImagingDevice = ImagingDevice
        self.sdb_structures = sdb_structures
        self.GrabFrameSettings = GrabFrameSettings

    @timing
    def initialize(self):
        self.microscope = self.SdbMicroscopeClient()
        self.connect()
        self.microscope.detector.type.value == "CBS"
        self.microscope.specimen.stage.set_default_coordinate_system(
            self.sdb_enums.CoordinateSystem.RAW
        )
        self.microscope.beams.electron_beam.unblank()

    @timing
    def close(self):
        self.disconnect()
        del self.microscope

    @timing
    def connect(self):
        self.microscope.connect(self.params["ip"])

    def disconnect(self):
        self.microscope.disconnect()

    @timing
    def prepare_acquisition(self):
        self.microscope.patterning.clear_patterns()
        self.auto_contrast_brightness(baseline=True)
        self.auto_focus(baseline=True)
        self.auto_stig(baseline=True)
        self.microscope.auto_functions.run_auto_lens_alignment()
        self.auto_stig()
        self.auto_focus()
        self.auto_contrast_brightness()

    @timing
    def auto_focus(self, baseline=False):
        baselineFocus = self.microscope.beams.electron_beam.working_distance.value
        AF_final_horizontal_field_width_focus = self.params[
            "AF_final_horizontal_field_width_focus"
        ]
        self.microscope.beams.electron_beam.horizontal_field_width.value = (
            AF_final_horizontal_field_width_focus
        )
        try:
            if baseline:
                self.microscope.auto_functions.run_auto_focus()
            else:
                af_settings = self.sdb_structures.RunAutoFocusSettings()
                af_settings.method = "Volumescope"
                af_settings.resolution = "1024x884"
                af_settings.dwell_time = self.params["slow_dwt_focus_ns"] * 1e-9
                af_settings.line_integration = 1
                af_settings.reduced_area = self.sdb_structures.Rectangle(
                    0.1, 0.1, 0.8, 0.02
                )
                af_settings.working_distance_step = 100e-9
                self.microscope.auto_functions.run_auto_focus(
                    af_settings
                )  # RunAutoFocusSettings structure item 11 does not have any value. proceeding with baseline focus
            newFocus = self.microscope.beams.electron_beam.working_distance.value
            if (newFocus * 1000) < 5.5:  # @YARON add explanation CHANGE TO 6.5?
                self.microscope.beams.electron_beam.working_distance.value = (
                    baselineFocus
                )
        except Exception as excp:
            warnings.warn(
                "Auto Focus failed " + str(excp) + " proceeding with baseline focus"
            )
            self.microscope.beams.electron_beam.working_distance.value = baselineFocus

    @timing
    def auto_contrast_brightness(self, baseline=False):
        if baseline:
            # need to run twice for @YARON add reason
            self.microscope.auto_functions.run_auto_cb()
            self.microscope.auto_functions.run_auto_cb()
        else:
            acb_settings = self.sdb_structures.RunAutoCbSettings()
            acb_settings.method = "MaxContrast"
            acb_settings.resolution = "512x442"
            acb_settings.calibrate_detector = True
            acb_settings.dwell_time = self.params["fast_dwt_contrast_ns"] * 1e-9
            acb_settings.number_of_frames = 1
            self.microscope.auto_functions.run_auto_cb(acb_settings)

    @timing
    def auto_stig(self, baseline=False):
        if baseline:
            self.microscope.auto_functions.run_auto_stigmator()
        else:
            try:
                AS_final_horizontal_field_width_stig = self.params[
                    "AS_final_horizontal_field_width_stig"
                ]
                as_settings = self.sdb_structures.RunAutoStigmatorSettings()
                as_settings.method = self.sdb_enums.AutoFunctionMethod.ONG_ET_AL_GENERAL
                as_settings.reduced_area = self.sdb_structures.Rectangle(
                    0.1, 0.1, 0.8, 0.8
                )
                self.microscope.beams.electron_beam.horizontal_field_width.value = (
                    AS_final_horizontal_field_width_stig
                )
                self.microscope.auto_functions.run_auto_stigmator(as_settings)
            except Exception as excp:
                warnings.warn("Auto Stig failed: " + str(excp))

    @timing
    def get_image(self, params):
        with time_block("prep_get_image"):
            params = copy.deepcopy(params)
            resolution = params["resolution"]
            pixel_size = params["pixel_size"]
            fov = (resolution[0] * pixel_size, resolution[1] * pixel_size)

            self.microscope.beams.electron_beam.scanning.resolution.value = (
                "%dx%d" % resolution
            )
            if "theta" in params.keys():
                self.microscope.beams.electron_beam.scanning.rotation.value = params[
                    "theta"
                ]
            self.microscope.imaging.set_active_view(1)
            self.microscope.imaging.set_active_device(self.ImagingDevice.ELECTRON_BEAM)
            self.microscope.beams.electron_beam.horizontal_field_width.value = fov[0]
            self.microscope.patterning.set_default_beam_type(
                self.sdb_enums.BeamType.ELECTRON
            )

            bit_depth = 16
        if "rescan_map" in params.keys():
            with time_block("prep_rescan"):
                rescan_map = params["rescan_map"]

                rescan_map = (rescan_map.astype(np.uint8) * 255)[:, :, None].repeat(
                    3, axis=2
                )

                # Simulator environment only serves images of ~1kx1k, so we will feed it an image of our desired shape
                if self.params["ip"] == "localhost":
                    from autoscript_sdb_microscope_client.structures import AdornedImage

                    tiff_path = (
                        Path(self.params["tempfile"]).parent.absolute()
                        / "tempfile.tiff"
                    )
                    tools.write_im(str(tiff_path), rescan_map[:, :, 0])
                    loaded_tiff = AdornedImage.load(tiff_path)
                    self.microscope.imaging.set_image(loaded_tiff)

                # image = self.microscope.imaging.get_image().data.copy() # not sure what this does

                self.microscope.patterning.clear_patterns()
            with time_block("write_rescan_map"):
                tools.write_im(self.params["tempfile"], rescan_map)
            with time_block("define_bitmap"):
                bpd = self.BitmapPatternDefinition.load(self.params["tempfile"])
            with time_block("create_pattern"):
                pattern = self.microscope.patterning.create_bitmap(
                    0, 0, fov[0], fov[1], params["dwell_time"], bpd
                )
                pattern.dwell_time = params["dwell_time"]
                pattern.pass_count = 1
                pattern.scan_type = self.sdb_enums.PatternScanType.RASTER
            with time_block("rescan"):
                self.microscope.patterning.run()

            with time_block("get_rescan"):
                image = (
                    self.microscope.imaging.get_image().data.copy()
                )  # Ask thermofisher if we can skip copy
                assert bit_depth == 16, "print only uint16 implemented"
                # self.microscope.patterning.clear_patterns() # only used for visualization on microscope computer?
        else:
            with time_block("prep_fastscan"):
                if "sleep" in self.params.keys():
                    time.sleep(30)
                settings = self.GrabFrameSettings(
                    resolution="%dx%d" % (resolution[0], resolution[1]),
                    dwell_time=params["dwell_time"],
                    bit_depth=bit_depth,
                )
            with time_block("fastscan"):
                image = self.microscope.imaging.grab_frame(settings).data
        if "invert" in params.keys() and params["invert"]:
            return np.iinfo(image.dtype).max - image
        else:
            return image

    @timing
    def move(self, x, y, z=None, r=None, t=None):
        if z is None or r is None or t is None:
            p = self.microscope.specimen.stage.current_position
            if z is None:
                z = p.z
            if r is None:
                r = p.r
            if t is None:
                t = p.t
        p2 = self.sdb_structures.StagePosition(
            x=x, y=y, z=z, r=r, t=t, coordinate_system="Raw"
        )
        self.microscope.specimen.stage.absolute_move(p2)
