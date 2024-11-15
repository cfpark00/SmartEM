import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import scipy.io as sio
from tqdm import tqdm
import time

from smartem import tools

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} args:[{args},{kw}] took:{te-ts:2.4f} sec")
        return result
    return wrap

class SmartEM:
    """
    SmartEM class to acquire with dynamic dwell time allocation.

    Args:
    microscope: Microscope, microscope object
    get_rescan_map: GetRescanMap, get_rescan_map object
    """

    def __init__(self, microscope, get_rescan_map):
        self.microscope = microscope
        self.get_rescan_map = get_rescan_map

    @timing
    def initialize(self):
        """
        Initialize the microscope and the get_rescan_map object.
        """
        self.microscope.initialize()
        self.get_rescan_map.initialize()

    @timing
    def prepare_acquisition(self):
        """
        Prepare the microscope for acquisition.
        """
        self.microscope.prepare_acquisition()

    @timing
    def close(self):
        """
        Close the microscope and the get_rescan_map object.
        """
        self.microscope.close()
        self.get_rescan_map.close()

    @timing
    def acquire(self, params):
        """
        Acquire with params, twice with fast and slow dwell times.

        Args:
        params: dict, parameters Required: fast_dwt, slow_dwt

        Returns:
        fast_em: np.ndarray, fast electron microscope image
        rescan_em: np.ndarray, rescan electron microscope image
        rescan_map: np.ndarray, rescan map
        additional: dict, additional information including figures or results from get_rescan_map
        """
        params = copy.deepcopy(params)
        params.update({"dwell_time": params["fast_dwt"]})
        fast_em = self.microscope.get_image(params=params)
        rescan_map, additional = self.get_rescan_map(fast_em)
        params.update({"dwell_time": params["slow_dwt"], "rescan_map": rescan_map})
        rescan_em = self.microscope.get_image(params=params)

        if "plot" in params and params["plot"]:
            fig = show_smart(
                fast_em,
                rescan_em,
                rescan_map,
                fast_dwt=params["fast_dwt"],
                slow_dwt=params["slow_dwt"],
            )
            additional["fig"] = fig
        return fast_em, rescan_em, rescan_map, additional

    @timing
    def acquire_to(self, save_dir, params):
        """
        Acquire with params and save to save_dir.

        Args:
        save_dir: str, directory to save
        params: dict, parameters Required: fast_dwt, slow_dwt, verbose

        Returns:
        None
        """
        fast_em, rescan_em, rescan_map, additional = self.acquire(params)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tools.write_im(os.path.join(save_dir, "fast_em.png"), fast_em)
        tools.write_im(os.path.join(save_dir, "rescan_em.png"), rescan_em)
        tools.write_im(
            os.path.join(save_dir, "rescan_map.png"),
            (rescan_map * 255).astype(np.uint8),
        )
        if "fig" in additional:
            additional["fig"].savefig(os.path.join(save_dir, "fig.png"))
        if "verbose" in params and params["verbose"] > 0:
            print(f"Saved to {save_dir}")

    @timing
    def acquire_grid(self, xyzrt, theta, nx, ny, dx, dy, params):
        """
        Acquire a grid of images with params.

        Args:
        xyzrt: np.ndarray, (5,) x, y, z, r, t
        theta: float, rotation angle
        nx: int, number of grid points in x
        ny: int, number of grid points in y
        dx: float, grid spacing in x
        dy: float, grid spacing in y
        params: dict, parameters Required: fast_dwt, slow_dwt

        Returns:
        return_dict: dict, dictionary of acquired images
        """
        params["theta"] = theta
        R = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
        x, y, z, r, t = xyzrt

        return_dict = {}
        for ix in range(nx):
            for iy in range(ny):
                coordinate = np.array([dx * ix, dy * iy]) @ R + np.array([x, y])
                self.microscope.move(x=coordinate[0], y=coordinate[1], z=z, r=r, t=t)
                fast_em, rescan_em, rescan_map, additional = self.acquire(params=params)
                return_dict[(ix, iy)] = {
                    "fast_em": fast_em,
                    "rescan_em": rescan_em,
                    "rescan_map": rescan_map,
                    "additional": additional,
                }
        return return_dict

    @timing
    def acquire_many_grids(self, coordinates, params, save_dir):
        """
        Acquire many grids with coordinates and params and save to save_dir.

        Args:
        coordinates: np.ndarray, (n, 5) x, y, z, r, t
        params: dict, imaging parameters
        save_dir: str, directory to save

        Returns:
        None
        """
        n_targets = len(coordinates)

        os.makedirs(save_dir, exist_ok=True)
        fast_fol = os.path.join(save_dir, "fast")
        os.makedirs(fast_fol, exist_ok=True)
        rescan_fol = os.path.join(save_dir, "rescan")
        os.makedirs(rescan_fol, exist_ok=True)
        rescan_map_fol = os.path.join(save_dir, "rescan_map")
        os.makedirs(rescan_map_fol, exist_ok=True)
        fast_mb_fol = os.path.join(save_dir, "fast_mb")
        os.makedirs(fast_mb_fol, exist_ok=True)

        for i in tqdm(
            range(n_targets), desc="Acquiring targets...", disable=not params["verbose"]
        ):
            fast_fol_ = os.path.join(fast_fol, "location_" + str(i).zfill(5))
            os.makedirs(fast_fol_, exist_ok=True)
            rescan_fol_ = os.path.join(rescan_fol, "location_" + str(i).zfill(5))
            os.makedirs(rescan_fol_, exist_ok=True)
            rescan_map_fol_ = os.path.join(
                rescan_map_fol, "location_" + str(i).zfill(5)
            )
            os.makedirs(rescan_map_fol_, exist_ok=True)
            fast_mb_fol_ = os.path.join(fast_mb_fol, "location_" + str(i).zfill(5))
            os.makedirs(fast_mb_fol_, exist_ok=True)

            xyzrt = coordinates[i]
            theta = params["scan_rotations"][i]

            fov = np.array(params["resolution"]) * params["pixel_size"]
            grid_results = self.acquire_grid(
                xyzrt=xyzrt,
                theta=theta,
                nx=2,  # hardcoded
                ny=2,
                dx=fov[0] * 0.8,
                dy=fov[1] * 0.8,
                params=params,
            )
            for key, value in grid_results.items():
                xi, yi = key
                tools.write_im(
                    os.path.join(
                        fast_fol_,
                        "location_" + str(i).zfill(5) + f"_xi_{xi}_yi_{yi}.png",
                    ),
                    value["fast_em"],
                )
                tools.write_im(
                    os.path.join(rescan_fol_, "location_" + str(i).zfill(5))
                    + f"_xi_{xi}_yi_{yi}.png",
                    value["rescan_em"],
                )
                tools.write_im(
                    os.path.join(rescan_map_fol_, "location_" + str(i).zfill(5))
                    + f"_xi_{xi}_yi_{yi}.png",
                    value["rescan_map"].astype(np.uint8) * 255,
                )
                tools.write_im(
                    os.path.join(fast_mb_fol_, "location_" + str(i).zfill(5))
                    + f"_xi_{xi}_yi_{yi}.png",
                    value["additional"]["fast_mb"],
                )

    @timing
    def acquire_many_grids_from_mat(self, target_mat, params, save_dir):
        """
        Acquire many grids from a .mat file and save to save_dir.

        Args:
        target_mat: str, path to the .mat file
        save_dir: str, directory to save

        Returns:
        None
        """
        target_mat = sio.loadmat(target_mat)
        n_targets = target_mat["nroftargets"].item()
        imaging = np.concatenate(target_mat["target"]["imaging"][:, 0], axis=0)
        stage_values = np.concatenate(
            target_mat["target"]["tempstagecoords"][:, 0], axis=0
        )

        params["brightness"] = np.concatenate(imaging["brightness"][:, 0], axis=0)[:, 0]
        params["contrasts"] = np.concatenate(imaging["contrast"][:, 0], axis=0)[:, 0]
        params["focus_val"] = np.concatenate(imaging["focus"][:, 0], axis=0)[:, 0]
        stigx = np.concatenate(imaging["stigx"][:, 0], axis=0)[:, 0]
        stigy = np.concatenate(imaging["stigy"][:, 0], axis=0)[:, 0]
        params["stigmations"] = np.stack([stigx, stigy], axis=1)
        params["scan_rotations"] = np.concatenate(
            target_mat["target"]["scanrot"][:, 0], axis=0
        )[:, 0]

        rs = np.concatenate(stage_values["rpos_rad"][:, 0], axis=0)[:, 0]
        ts = np.concatenate(stage_values["tpos_rad"][:, 0], axis=0)[:, 0]
        xs = np.concatenate(stage_values["xpos_m"][:, 0], axis=0)[:, 0]
        ys = np.concatenate(stage_values["ypos_m"][:, 0], axis=0)[:, 0]
        zs = np.concatenate(stage_values["zpos_m"][:, 0], axis=0)[:, 0]

        coordinates = np.stack([xs, ys, zs, rs, ts], axis=1)
        assert len(coordinates) == n_targets

        # coordinates, imaging_params are the parsed outputs

        self.acquire_many_grids(
            coordinates=coordinates, params=params, save_dir=save_dir
        )

    def __str__(self):
        return (
            "SmartEM with microscope:\n"
            + str(self.microscope)
            + "\nand get_rescan_map:\n"
            + str(self.get_rescan_map)
        )


def show_smart(fast_em, slow_em, rescan_map, fast_dwt, slow_dwt):
    """
    Make a figure to show the results of SmartEM.

    Args:
    fast_em: np.ndarray, fast electron microscope image
    slow_em: np.ndarray, slow electron microscope image
    rescan_map: np.ndarray, rescan map
    fast_dwt: float, fast dwell time
    slow_dwt: float, slow dwell time

    Returns:
    fig: plt.Figure, figure
    """
    fig = plt.figure(figsize=(20, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(fast_em, interpolation="none", cmap="gray")
    plt.title(f"fast_em, dwell_time = {fast_dwt*1e9:.0f} ns")
    plt.subplot(2, 3, 2)
    plt.imshow(rescan_map, interpolation="none", cmap="gray")
    plt.title("rescan_map")
    plt.subplot(2, 3, 3)
    plt.imshow(slow_em, interpolation="none", cmap="gray")
    plt.title(f"slow_em, dwell_time = {slow_dwt*1e9:.0f} ns")
    plt.subplot(2, 3, 4)
    merged_em = fast_em.copy()
    merged_em[rescan_map] = slow_em[rescan_map]
    plt.imshow(merged_em, interpolation="none", cmap="gray")
    plt.title("merged_em")
    plt.subplot(2, 3, 5)
    plt.imshow(merged_em - fast_em, interpolation="none", cmap="gray")
    plt.title(f"merged_em - fast_em")
    return fig
