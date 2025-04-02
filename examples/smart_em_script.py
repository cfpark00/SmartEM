# Example script using SmartEM
# See the main function at the end of the file for the command line arguments
# Author: Core Francisco Park, corefranciscopark@g.harvard.edu
############################################

# basic imports
import os
import numpy as np
import json
from pathlib import Path

# add the base SmartEM path to the python path
from smartem.smartem import SmartEM
from smartem.smartem_par import SmartEMPar
from smartem.online import microscope, get_rescan_maps


###########################################
# write functions simply handling different cases of the examples

# python examples\smart_em_script.py --get-rescan-map-type membrane_errors --target-mat examples\default_imaging_params_pres.json --params-path examples\default_smartem_params_pres.json
# python examples\smart_em_script.py --get-rescan-map-type membrane_errors --target-mat examples\w03_imaging_params_single.json --params-path examples\default_smartem_params_pres.json --microscope-type verios

default_target_mat = "D:\\Users\\Lab\\Documents\\SmartEM\\data\\Mouse_NK1\\wafer_calibration\\w03_1mm_nov20.mat"


def get_microscope(microscope_type):
    """
    Get the microscope object

    Args:
    microscope_type: str, type of microscope to use

    Returns:
    microscope: microscope object
    """
    if microscope_type == "verios":
        # This is the microscope used for the SmartEM paper
        params = {"ip": "192.168.0.1"}  # online mode (microscope active)
        #params = {"ip":  "localhost"}  # offline mode
        my_microscope = microscope.ThermoFisherVerios(params=params)
    elif microscope_type == "fake":
        # This is a fake microscope that generates random images
        params = {"W": 1024, "H": 1024, "dtype": np.uint16}
        my_microscope = microscope.FakeRandomMicroscope(params=params)
    elif microscope_type == "fakedata":
        # This is a fake microscope that uses example data
        params = {
            "images_ns": {
                50: "./examples/data/example1/loc_001_dwell_00050ns_00002_param_001_yi_1_xi_1_reg.png",
                75: "./examples/data/example1/loc_001_dwell_00050ns_00002_param_001_yi_1_xi_1_reg.png",
                100: "./examples/data/example1/loc_001_dwell_00100ns_00004_param_001_yi_1_xi_1_reg.png",
                200: "./examples/data/example1/loc_001_dwell_00200ns_00007_param_001_yi_1_xi_1_reg.png",
                500: "./examples/data/example1/loc_001_dwell_00500ns_00010_param_001_yi_1_xi_1_reg.png",
                800: "./examples/data/example1/loc_001_dwell_00500ns_00010_param_001_yi_1_xi_1_reg.png",
                1200: "./examples/data/example1/loc_001_dwell_01200ns_00014_param_001_yi_1_xi_1_reg.png",
            }
        }
        my_microscope = microscope.FakeDataMicroscope(params=params, sleep=False, pad_images=True)
    else:
        raise ValueError("Unknown microscope type")
    return my_microscope


def get_get_rescan_map(
    get_rescan_map_type,
):  # Not a typo, we are getting get_rescan_map object
    """
    Get the get_rescan_map object

    Args:
    get_rescan_map_type: str, type of get_rescan_map to use

    Returns:
    get_rescan_map: get_rescan_map object
    """
    if get_rescan_map_type == "test":
        # This is a test get_rescan_map that returns a half image
        params = {"type": "half", "fraction": 0.5}
        get_rescan_map = get_rescan_maps.GetRescanMapTest(params=params)
    elif get_rescan_map_type == "membrane_errors":
        # This is the get_rescan_map using ML
        params = {
            "em2mb_net": "./pretrained_models/em2mb_best.pth",
            "error_net": "./pretrained_models/error_best.pth",
            "device": "auto",
            "pad": 40,
            "rescan_p_thres": 0.1,
            "rescan_ratio": 0.1,  # add a number to force a specific rescan ratio
            "search_step": 0.01,
            "do_clahe": True,
        }
        get_rescan_map = get_rescan_maps.GetRescanMapMembraneErrors(params=params)
    else:
        raise ValueError("Unknown get_rescan_map method")
    return get_rescan_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--microscope-type", type=str, default="fakedata")
    parser.add_argument(
        "--params-path", type=str, default="examples/default_smartem_params.json"
    )
    parser.add_argument(
        "--get-rescan-map-type", type=str, default="test"
    )  # membrane_errors")
    parser.add_argument(
        "--target-mat",
        type=str,
        default=default_target_mat,
    )
    parser.add_argument("--save-dir", type=str, default="./data/test_94")
    args = parser.parse_args()

    # Check the arguments
    microscope_type = args.microscope_type
    assert microscope_type in [
        "verios",
        "fake",
        "fakedata",
    ], "Unknown microscope type, choose from 'verios', 'fake', 'fakedata'"

    params_path = args.params_path
    assert os.path.exists(params_path), f"params_path {params_path} does not exist"

    get_rescan_map_type = args.get_rescan_map_type
    assert get_rescan_map_type in [
        "test",
        "membrane_errors",
    ], "Unknown get_rescan_map type, choose from 'test', 'membrane_errors'"
    target_mat = args.target_mat
    if microscope_type == "verios":
        assert os.path.exists(target_mat), f"target_mat {target_mat} does not exist"
    save_dir = args.save_dir

    # Get the microscope and get_rescan_map objects
    my_microscope = get_microscope(microscope_type)
    with open(params_path, "r") as f:
        params = json.load(f)
        if "resolution" in params:
            params["resolution"] = tuple(params["resolution"])
    get_rescan_map = get_get_rescan_map(get_rescan_map_type)

    # Initialize Microscope
    print("Initializing Microscope.....")
    serial = True
    if serial:
        print("Serial mode.....")
        my_smart_em = SmartEM(microscope=my_microscope, get_rescan_map=get_rescan_map)
    else:
        print("Parallel mode.....")
        my_smart_em = SmartEMPar(microscope=my_microscope, get_rescan_map=get_rescan_map)
    my_smart_em.initialize()
    print("Microscope:", my_smart_em)
    print()

    print("Prepare acquisition.....")
    my_smart_em.prepare_acquisition()

    print("Acquiring...")
    # Set some parameters
    if microscope_type == "verios":
        target_mat_type = Path(target_mat).suffix
        if target_mat_type == ".mat":
            my_smart_em.acquire_many_grids_from_mat(
                target_mat=target_mat, save_dir=save_dir, params=params
            )
        elif target_mat_type == ".json":
            with open(target_mat, "r") as f:
                params_imaging = json.load(f)
            params.update(params_imaging)
            my_smart_em.acquire_many_grids(
                coordinates=params["coordinates"], params=params, save_dir=save_dir
            )
    elif microscope_type == "fake":
        my_smart_em.acquire_to(save_dir=save_dir, params=params)
    elif microscope_type == "fakedata":
        if target_mat == default_target_mat:
            my_smart_em.acquire_to(save_dir=save_dir, params=params)
        else:
            with open(target_mat, "r") as f:
                params_imaging = json.load(f)
            params.update(params_imaging)
            my_smart_em.acquire_many_grids(
                coordinates=params["coordinates"], params=params, save_dir=save_dir
            )

    print("Closing.....")
    my_smart_em.close()

    print("Run Successful!")
