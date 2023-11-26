import abc

### package imports ###
import numpy as np
import os

from src import tools


class BaseMicroscope(metaclass=abc.ABCMeta):
    def __init__(self):
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

    def get_image(self, params):
        W = params["W"] if "W" in params else 1024
        H = params["H"] if "H" in params else 1024
        dtype = params["dtype"] if "dtype" in params else np.uint16
        image = np.random.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, (W, H), dtype=dtype
        )
        if "rescan_map" in params.keys():
            image[~params["rescan_map"]] = np.iinfo(dtype).min
            return image
        else:
            return image

    def move(self, params):
        raise NotImplementedError("No move implemented for this microscope")

    def initialize(self):
        pass

    def close(self):
        pass


class FakeDataMicroscope(BaseMicroscope):
    default_params = {"images_ns": {}}

    def __init__(self, params=None):
        super().__init__()
        self.params = self.default_params
        if params is not None:
            self.params.update(params)
        for key in self.params["images_ns"].keys():
            assert os.path.exists(
                self.params["images_ns"][key]
            ), f"File {self.params['images_ns'][key]} does not exist"

    def get_image(self, params):
        dwt = params["dwell_time"]
        dwt_ns = int(dwt * 1e9)
        if dwt_ns not in self.params["images_ns"].keys():
            raise ValueError(f"No data for dwell time {dwt_ns} ns")
        file_path = self.params["images_ns"][dwt_ns]
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        return tools.load_im(file_path)

    def move(self, params):
        raise NotImplementedError("No move implemented for this microscope")

    def initialize(self):
        pass

    def close(self):
        pass


class ThermoFisherVerios(BaseMicroscope):
    default_params = {"tempfile": "./tempfile.bmp"}

    def __init__(self, params=None):
        super().__init__()
        self.params = self.default_params
        if params is not None:
            self.params.update(params)
        assert "ip" in self.params.keys(), "ip address of microscope must be provided"
        assert os.path.exists(
            os.path.split(self.params["tempfile"])[0]
        ), "tempfile folder does not exist"

        from autoscript_sdb_microscope_client import SdbMicroscopeClient
        import autoscript_sdb_microscope_client.enumerations as sdb_enums
        import autoscript_sdb_microscope_client.sdb_microscope_client.BitmapPatternDefinition as BitmapPatternDefinition
        from autoscript_sdb_microscope_client.enumerations import ImagingDevice
        from autoscript_sdb_microscope_client.structures import GrabFrameSettings

        self.SdbMicroscopeClient = SdbMicroscopeClient
        self.sdb_enums = sdb_enums
        self.BitmapPatternDefinition = BitmapPatternDefinition
        self.ImagingDevice = ImagingDevice
        self.GrabFrameSettings = GrabFrameSettings

    def initialize(self):
        self.microscope = self.SdbMicroscopeClient()
        self.connect()
        self.microscope.detector.type.value == "CBS"
        self.microscope.specimen.stage.set_default_coordinate_system(
            self.sdb_enums.CoordinateSystem.RAW
        )

    def close(self):
        self.disconnect()
        del self.microscope

    def connect(self):
        self.microscope.connect(self.params["ip"])

    def disconnect(self):
        self.microscope.disconnect()

    def get_image(self, params):
        high_res = (2048, 1768)
        fov = (high_res[0] * 4.0e-9, high_res[1] * 4.0e-9)

        self.microscope.beams.electron_beam.scanning.resolution.value = (
            "%dx%d" % high_res
        )
        self.microscope.imaging.set_active_view(1)
        self.microscope.imaging.set_active_device(self.ImagingDevice.ELECTRON_BEAM)
        self.microscope.beams.electron_beam.horizontal_field_width.value = fov[0]

        bit_depth = 16
        if "rescan_map" in params.keys():
            rescan_map = params["rescan_map"]

            rescan_map = (rescan_map.astype(np.uint8) * 255)[:, :, None].repeat(
                3, axis=2
            )
            self.microscope.patterning.clear_patterns()
            tools.write_im(self.params["tempfile"], rescan_map)
            bpd = self.BitmapPatternDefinition.load(self.params["tempfile"])
            pattern = self.microscope.patterning.create_bitmap(
                0, 0, fov[0], fov[1], params["dwell_time"], bpd
            )
            pattern.dwell_time = params["dwell_time"]
            pattern.pass_count = 1
            pattern.scan_type = self.sdb_enums.PatternScanType.RASTER
            self.microscope.beams.electron_beam.unblank()
            self.microscope.patterning.run()

            image = (
                self.microscope.imaging.get_image().data.copy()
            )  # Ask thermofisher if we can skip copy
            assert bit_depth == 16, "print only uint16 implemented"
            self.microscope.patterning.clear_patterns()
        else:
            settings = self.GrabFrameSettings(
                resolution="%dx%d" % (high_res[0], high_res[1]),
                dwell_time=params["dwell_time"],
                bit_depth=bit_depth,
            )
            image = self.microscope.imaging.grab_frame(settings).data
        return image

    def move(self, params):
        raise NotImplementedError("No move implemented")
