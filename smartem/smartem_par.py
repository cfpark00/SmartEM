import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import scipy.io as sio

from threading import Event, Thread
import time

from smartem import tools

fast_ims = []
rescan_masks = []


def acquire(locs, fast_ims, sleep_time):
    for loc in locs:
        fast_ims.append(loc)
        time.sleep(sleep_time)


def compute(locs, fast_ims, rescan_masks, sleep_time):
    counter = 0
    while counter < len(locs):
        if len(fast_ims) > counter:
            im = fast_ims[counter]
            time.sleep(sleep_time)
            rescan_masks.append(im + 1)
            counter += 1


class par_test:
    def __init__(self):
        pass

    def run(self, locs, sleep_a, sleep_b):
        a = Thread(target=acquire, args=(locs, fast_ims, sleep_a))
        b = Thread(
            target=compute,
            args=(
                locs,
                fast_ims,
                rescan_masks,
                sleep_b,
            ),
        )

        a.start()
        b.start()

        a.join()
        b.join()
        return rescan_masks


def acquire_grid_fast(microscope, xyzrt, theta, nx, ny, dx, dy, params, fast_ems):
    R = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
    x, y, z, r, t = xyzrt

    for ix in range(nx):
        for iy in range(ny):
            coordinate = np.array([dx * ix, dy * iy]) @ R + np.array([x, y])
            microscope.move(x=coordinate[0], y=coordinate[1], z=z, r=r, t=t)
            params = copy.deepcopy(params)
            params.update({"dwell_time": params["fast_dwt"]})
            print(f"acquiring image: {ix*nx+iy}")
            fast_em = microscope.get_image(params=params)
            fast_ems.append(fast_em)


def compute_grid(get_rescan_map, nx, ny, fast_ems, rescan_maps, additionals):
    n_tiles = nx * ny
    counter = 0

    while counter < n_tiles:
        if len(fast_ems) > counter:
            fast_em = fast_ems[counter]
            print(f"computing rescan map: {counter}")
            rescan_map, additional = get_rescan_map(fast_em)
            rescan_maps.append(rescan_map)
            additionals.append(additional)
            counter += 1


class SmartEMPar:
    def __init__(self, microscope, get_rescan_map):
        self.microscope = microscope
        self.get_rescan_map = get_rescan_map

    def initialize(self):
        """
        Initialize the microscope and the get_rescan_map object.
        """
        self.microscope.initialize()
        self.get_rescan_map.initialize()

    def prepare_acquisition(self):
        """
        Prepare the microscope for acquisition.
        """
        self.microscope.prepare_acquisition()

    def close(self):
        """
        Close the microscope and the get_rescan_map object.
        """
        self.microscope.close()
        self.get_rescan_map.close()

    def acquire_grid_rescan(self, xyzrt, theta, nx, ny, dx, dy, params, rescan_maps):
        R = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
        x, y, z, r, t = xyzrt

        rescan_ems = []

        for ix in range(nx):
            for iy in range(ny):
                rescan_map = rescan_maps[ix * nx + iy]
                coordinate = np.array([dx * ix, dy * iy]) @ R + np.array([x, y])
                self.microscope.move(x=coordinate[0], y=coordinate[1], z=z, r=r, t=t)
                params = copy.deepcopy(params)
                params.update(
                    {"dwell_time": params["slow_dwt"], "rescan_map": rescan_map}
                )
                rescan_em = self.microscope.get_image(params=params)
                rescan_ems.append(rescan_em)

        return rescan_ems

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

    def acquire_grid(self, xyzrt, theta, nx, ny, dx, dy, params):
        params["theta"] = theta

        # compute rescan masks in parallel with acquisition
        fast_ems = []
        rescan_maps = []
        additionals = []

        a = Thread(
            target=acquire_grid_fast,
            args=(self.microscope, xyzrt, theta, nx, ny, dx, dy, params, fast_ems),
        )
        b = Thread(
            target=compute_grid,
            args=(self.get_rescan_map, nx, ny, fast_ems, rescan_maps, additionals),
        )

        a.start()
        b.start()
        a.join()
        b.join()

        # rescan
        rescan_ems = self.acquire_grid_rescan(
            xyzrt, theta, nx, ny, dx, dy, params, rescan_maps
        )

        if "verbose" in params and params["verbose"] > 0:
            print(f"Acquired fast_em, rescan_em, rescan_map")

        return_dict = {}
        for ix in range(nx):
            for iy in range(ny):
                idx = ix * nx + iy
                fast_em, rescan_em, rescan_map, additional = (
                    fast_ems[idx],
                    rescan_ems[idx],
                    rescan_maps[idx],
                    additionals[idx],
                )

                if "plot" in params and params["plot"]:
                    fig = show_smart(
                        fast_em,
                        rescan_em,
                        rescan_map,
                        fast_dwt=params["fast_dwt"],
                        slow_dwt=params["slow_dwt"],
                    )
                    additional["fig"] = fig

                return_dict[(ix, iy)] = {
                    "fast_em": fast_em,
                    "rescan_em": rescan_em,
                    "rescan_map": rescan_map,
                    "additional": additional,
                }

        return return_dict

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

        for i in range(n_targets):
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
                nx=2,  # hard coded
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
