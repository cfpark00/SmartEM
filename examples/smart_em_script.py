#Example script using SmartEM
import os

import numpy as np

#add the base SmartEM path to the python path
import sys
sys.path.append('..')

from src import smartem,tools
from src.online import microscope,get_rescan_maps

microscope_type="verios"
get_rescan_map_type="membrane_errors"

target_mat = "D:\\Users\\Lab\\Documents\\SmartEM\\data\\Mouse_NK1\\wafer_calibration\\w03_1mm_nov20.mat"
save_directory="./data/test_94"

###########################################
if microscope_type == "verios":
    params = {"ip": "192.168.0.1"}
    my_microscope = microscope.ThermoFisherVerios(params=params)
elif microscope_type == "fake":
    params = {"W": 1024, "H": 1024, "dtype": np.uint16}
    my_microscope = microscope.FakeRandomMicroscope(params=params)
elif microscope_type == "fake_data":
    params = {
        "images_ns": {
            50: "../examples/data/example1/loc_001_dwell_00050ns_00002_param_001_yi_1_xi_1_reg.png",
            100: "../examples/data/example1/loc_001_dwell_00100ns_00004_param_001_yi_1_xi_1_reg.png",
            200: "../examples/data/example1/loc_001_dwell_00200ns_00007_param_001_yi_1_xi_1_reg.png",
            500: "../examples/data/example1/loc_001_dwell_00500ns_00010_param_001_yi_1_xi_1_reg.png",
            1200: "../examples/data/example1/loc_001_dwell_01200ns_00014_param_001_yi_1_xi_1_reg.png",
        }
    }
    my_microscope = microscope.FakeDataMicroscope(params=params)
else:
    raise ValueError("Unknown microscope type")

if get_rescan_map_type == "test":
    params = {"type": "half", "fraction": 0.5}
    get_rescan_map = get_rescan_maps.GetRescanMapTest(params=params)
elif get_rescan_map_type == "membrane_errors":
    params = {
        "em2mb_net": "../pretrained_models/em2mb_best.pth",
        "error_net": "../pretrained_models/error_best.pth",
        "device": "auto",
        "pad": 40,
        "rescan_p_thres": 0.1,
        "rescan_ratio":None,#add a number to force a specific rescan ratio
        "search_step": 0.01,
        "do_clahe": True,
    }

    get_rescan_map = get_rescan_maps.GetRescanMapMembraneErrors(params=params)
else:
    raise ValueError("Unknown get_rescan_map method")

###### OPERATE MICROSCOPE
my_smart_em = smartem.SmartEM(
    microscope=my_microscope, get_rescan_map=get_rescan_map
)


my_smart_em.initialize()
print(my_smart_em)

my_smart_em.prepare_acquisition()

my_smart_em.acquire_many_grids_from_mat(target_mat=target_mat,save_directory=save_directory)

my_smart_em.close()
