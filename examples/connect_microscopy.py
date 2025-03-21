from autoscript_sdb_microscope_client import SdbMicroscopeClient
import autoscript_sdb_microscope_client.enumerations as sdb_enums

microscope = SdbMicroscopeClient()
microscope.connect("192.168.0.1")

microscope.detector.type.value == "CBS"
microscope.specimen.stage.set_default_coordinate_system(sdb_enums.CoordinateSystem.RAW)