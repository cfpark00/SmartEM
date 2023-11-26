import matplotlib.pyplot as plt
import numpy as np
import os

from src import tools


class SmartEM:
    def __init__(self, microscope, get_rescan_map):
        self.microscope = microscope
        self.get_rescan_map = get_rescan_map

    def initialize(self):
        self.microscope.initialize()
        self.get_rescan_map.initialize()

    def close(self):
        self.microscope.close()
        self.get_rescan_map.close()

    def acquire(self, params=None):
        fast_dwt = params["fast_dwt"]
        slow_dwt = params["slow_dwt"]

        fast_em = self.microscope.get_image({"dwell_time": fast_dwt})
        rescan_map = self.get_rescan_map(fast_em)
        slow_em = self.microscope.get_image(
            {"dwell_time": slow_dwt, "rescan_map": rescan_map}
        )

        if "plot" in params and params["plot"]:
            show_smart(fast_em, slow_em, rescan_map, fast_dwt, slow_dwt)
        return fast_em, slow_em, rescan_map

    def acquire_to(self, fol_path, params=None):
        fast_em, slow_em, rescan_map = self.acquire(params)
        if not os.path.exists(fol_path):
            os.mkdir(fol_path)
        tools.write_im(os.path.join(fol_path, "fast_em.png"), fast_em)
        tools.write_im(os.path.join(fol_path, "slow_em.png"), slow_em)
        tools.write_im(
            os.path.join(fol_path, "rescan_map.png"),
            (rescan_map * 255).astype(np.uint8),
        )

    def __str__(self):
        return (
            "SmartEM with microscope:\n"
            + str(self.microscope)
            + "\nand get_rescan_map:\n"
            + str(self.get_rescan_map)
        )


def show_smart(fast_em, slow_em, rescan_map, fast_dwt, slow_dwt):
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(fast_em, interpolation="none")
    plt.title(f"fast_em, dwell_time = {fast_dwt*1e9:.0f} ns")
    plt.subplot(2, 3, 2)
    plt.imshow(rescan_map, interpolation="none")
    plt.title("rescan_map")
    plt.subplot(2, 3, 3)
    plt.imshow(slow_em, interpolation="none")
    plt.title(f"slow_em, dwell_time = {slow_dwt*1e9:.0f} ns")
    plt.subplot(2, 3, 4)
    merged_em = fast_em.copy()
    merged_em[rescan_map] = slow_em[rescan_map]
    plt.imshow(merged_em, interpolation="none")
    plt.title("merged_em")
    plt.subplot(2, 3, 5)
    plt.imshow(merged_em - fast_em, interpolation="none")
    plt.title(f"merged_em - fast_em")
    plt.show()
