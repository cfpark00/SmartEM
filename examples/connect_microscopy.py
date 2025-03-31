from autoscript_sdb_microscope_client import SdbMicroscopeClient
import autoscript_sdb_microscope_client.enumerations as sdb_enums
import autoscript_sdb_microscope_client.sdb_microscope_client as sdb_microscope_client


microscope = SdbMicroscopeClient()
microscope.connect()#"192.168.0.1")

microscope.detector.type.value == "CBS"
microscope.specimen.stage.set_default_coordinate_system(sdb_enums.CoordinateSystem.RAW)


microscope.specimen.stage.set_default_coordinate_system(
    sdb_enums.CoordinateSystem.RAW
)

bmp_path = "D:\\Users\\Lab\\Documents\\SmartEM\\athey\\SmartEM\\tempfile.bmp"
bpd = sdb_microscope_client.BitmapPatternDefinition.load(bmp_path)

microscope.patterning.create_bitmap(0, 0, 8.192e-06, 7.072e-06, 8e-07, bpd)