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
