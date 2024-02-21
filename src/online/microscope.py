import abc
import copy

### package imports ###
import numpy as np
import os
import warnings

from src import tools

import copy

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
    default_params = {"tempfile": "./tempfile.bmp",
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

        self.sdb_client=sdb_client
        self.SdbMicroscopeClient = SdbMicroscopeClient
        self.sdb_enums = sdb_enums
        self.BitmapPatternDefinition = sdb_microscope_client.BitmapPatternDefinition
        self.ImagingDevice = ImagingDevice
        self.sdb_structures = sdb_structures
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

    def prepare_acquisition(self):
        self.auto_contrast_brightness(baseline=True)
        self.auto_focus(baseline=True)
        self.microscope.auto_functions.run_auto_lens_alignment()
        self.auto_stig()
        self.auto_focus()
        self.auto_contrast_brightness()

    def auto_focus(self,baseline=False):
        baselineFocus = self.microscope.beams.electron_beam.working_distance.value
        AF_final_horizontal_field_width_focus=self.params["AF_final_horizontal_field_width_focus"]
        self.microscope.beams.electron_beam.horizontal_field_width.value = AF_final_horizontal_field_width_focus
        try:
            if baseline:
                self.microscope.auto_functions.run_auto_focus()
            else:
                af_settings = self.sdb_structures.RunAutoFocusSettings()
                af_settings.method = "Volumescope"
                af_settings.resolution = "1024x884"
                af_settings.dwell_time = self.params["slow_dwt_focus_ns"] * 1e-9
                af_settings.line_integration = 1
                af_settings.reduced_area = self.sdb_structures.Rectangle(0.1, 0.1, 0.8, 0.02)
                af_settings.working_distance_step = 100e-9
                self.microscope.auto_functions.run_auto_focus(af_settings)
            newFocus = self.microscope.beams.electron_beam.working_distance.value
            if (newFocus * 1000) < 5.5 : #@YARON add explanation
                self.microscope.beams.electron_beam.working_distance.value = baselineFocus
        except Exception as excp:
            warnings.warn("Auto Focus failed "+str(excp)+" proceeding with baseline focus")
            self.microscope.beams.electron_beam.working_distance.value = baselineFocus

    def auto_contrast_brightness(self,baseline=False):
        if baseline:
            #need to run twice for @YARON add reason
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

    def auto_stig(self):
        try:
            AS_final_horizontal_field_width_stig=self.params["AS_final_horizontal_field_width_stig"]
            as_settings = self.sdb_structures.RunAutoStigmatorSettings()
            as_settings.method = self.sdb_enums.AutoFunctionMethod.ONG_ET_AL_GENERAL
            as_settings.reduced_area = self.sdb_structures.Rectangle(0.1, 0.1, 0.8, 0.8)
            self.microscope.beams.electron_beam.horizontal_field_width.value = AS_final_horizontal_field_width_stig;
            self.microscope.auto_functions.run_auto_stigmator(as_settings)
        except Exception as excp:
            warnings.warn("Auto Stig failed: "+str(excp))

    def get_image(self, params):
        params=copy.deepcopy(params)
        resolution=params["resolution"]
        pixel_size=params["pixel_size"]
        fov = (resolution[0] * pixel_size, resolution[1] * pixel_size)

        self.microscope.beams.electron_beam.scanning.resolution.value = (
            "%dx%d" % resolution
        )
        if "theta" in params.keys():
            self.microscope.beams.electron_beam.scanning.rotation.value=params["theta"]
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
                resolution="%dx%d" % (resolution[0], resolution[1]),
                dwell_time=params["dwell_time"],
                bit_depth=bit_depth,
            )
            image = self.microscope.imaging.grab_frame(settings).data
        if "invert" in params.keys() and params["invert"]:
            return np.iinfo(image.dtype).max-image
        else:
            return image

    """
    def get_image_grid(self,start_x,start_y,start_z,dx,dy,):

        shift = 0.8 * fov;
        v0 = direction * ([rasterX(yi, xi)(rasterY(yi, xi))] - 1). * shift;
        v0R = v0 * R;

        pxyi = stagePositions(iN, 1:2)+v0R;

        % p0.x = p0.x + v0R(1);
        % p0.y = p0.y + v0R(2);

        if norm(pxyi - pxy_last) > 50e-9 % % % if location changed by more than 50 nm, apply a move
        p1.x = pxyi(1);
        p1.y = pxyi(2);
        try
            tic;
            microscope.specimen.stage.absolute_move(p1);
            toc
        catch
        warning('failed to move the stage')
        keyboard
        continue

    end
    """

    def move(self,x,y,z=None,r=None,t=None):
        if z is None or r is None or t is None:
            p=self.microscope.specimen.stage.current_position
            if z is None:
                z=p.z
            if r is None:
                r=p.r
            if t is None:
                t=p.t
        p2=self.sdb_structures.StagePosition(x=x, y=y, z=z, r=r, t=t, coordinate_system="Raw")
        self.microscope.specimen.stage.absolute_move(p2)
