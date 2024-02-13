#Example script using SmartEM

import numpy as np
import scipy.io as sio

#add the base SmartEM path to the python path
import sys
sys.path.append('..')

from src import smartem,tools
from src.online import microscope,get_rescan_maps

microscope_type="verios"

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

get_rescan_map_type="membrane_errors"

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


###### LOAD MATFILE
target_mat = "D:\\Users\\Lab\\Documents\\SmartEM\\data\\Mouse_NK1\\wafer_calibration\\w03_1mm_nov20.mat"
target_mat=sio.loadmat(target_mat)
n_targets=target_mat["nroftargets"].item()
imaging=np.concatenate(target_mat["target"]["imaging"][:,0],axis=0)
stage_values = np.concatenate(target_mat["target"]["tempstagecoords"][:,0],axis=0)

imaging_params={}
imaging_params["brightness"]=np.concatenate(imaging["brightness"][:,0],axis=0)[:,0]
imaging_params["contrasts"]=np.concatenate(imaging["contrast"][:,0],axis=0)[:,0]
imaging_params["focus_val"]=np.concatenate(imaging["focus"][:,0],axis=0)[:,0]
stigx=np.concatenate(imaging["stigx"][:,0],axis=0)[:,0]
stigy=np.concatenate(imaging["stigy"][:,0],axis=0)[:,0]
imaging_params["stigmations"]=np.stack([stigx,stigy],axis=1)
imaging_params["scan_rotations"]=np.concatenate(target_mat["target"]["scanrot"][:,0],axis=0)[:,0]

rs=np.concatenate(stage_values["rpos_rad"][:,0],axis=0)[:,0]
ts=np.concatenate(stage_values["tpos_rad"][:,0],axis=0)[:,0]
xs=np.concatenate(stage_values["xpos_m"][:,0],axis=0)[:,0]
ys=np.concatenate(stage_values["ypos_m"][:,0],axis=0)[:,0]
zs=np.concatenate(stage_values["zpos_m"][:,0],axis=0)[:,0]
coordinates=np.stack([xs,ys,zs,rs,ts],axis=1)

###### OPERATE MICROSCOPE
my_smart_em = smartem.SmartEM(
    microscope=my_microscope, get_rescan_map=get_rescan_map
)

my_smart_em.initialize()
print(my_smart_em)

if False:
    for i in range(len(xs)):
        xyzrt = [xs[i], ys[i], zs[i], rs[i], ts[i]]
        theta=imaging_params["scan_rotations"][i]

        params = {"fast_dwt": 50e-9, "slow_dwt": 500e-9, "plot": False, "invert": True, "theta": theta, "resolution": (2048, 1768),
                  "pixel_size": 4.0e-9}
        fov=np.array(params["resolution"])*params["pixel_size"]
        my_smart_em.acquire_grid(xyzrt=xyzrt, theta=theta, nx=2, ny=2, dx=fov[0]*0.8, dy=fov[1]*0.8, params=params)
        break
else:
    params = {"fast_dwt": 50e-9, "slow_dwt": 500e-9, "plot": False, "invert": True, "resolution":(2048, 1768), "pixel_size": 4.0e-9,"theta":0.}
    fast_em, slow_em, rescan_map, additional = my_smart_em.acquire(params=params)
    #not making a new params makes an memory error somewhere.
    fast_em, slow_em, rescan_map, additional = my_smart_em.acquire(params=params)

tools.write_im("fast_em.png",fast_em)
tools.write_im("slow_em.png",slow_em)
tools.write_im("rescan_map.png",rescan_map.astype(np.uint8)*255)
tools.write_im("fast_mb.png",additional["fast_mb"])
tools.write_im("error_prob.png",(additional["error_prob"]*255).astype(np.uint8))
my_smart_em.close()