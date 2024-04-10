import abc

#### import packages first to avoid overhead
import torch
import skimage.morphology as skmorph
import os
import numpy as np
import warnings
import cv2

from smartem import tools
# from .models import UNet
from FM_work.SmartEM.smartem.segmentation import UNet

class GetRescanMap(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __call__(self, fast_em):
        return self.get_rescan_map(fast_em)

    @abc.abstractmethod
    def get_rescan_map(self, fast_em):
        """
        Must return a boolean array of the same shape as fast_em and a dictionary
        """
        pass

    @abc.abstractmethod
    def initialize(self):
        """
        Initialize any objects with overhead
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        Any cleanup if necessary
        """
        pass


class GetRescanMapTest(GetRescanMap):
    default_params = {
        "type": "half",
        "fraction": 0.5,
    }
    available_types = ["half", "random", "threshold"]

    def __init__(self, params=None):
        super().__init__()
        self.params = self.default_params
        if params is not None:
            self.params.update(params)
        assert self.params["type"] in self.available_types

    def get_rescan_map(self, fast_em):
        if self.params["type"] == "half":
            mask = np.zeros_like(fast_em, dtype=bool)
            mask[: mask.shape[0] // 2] = 1
            return mask, {}
        elif self.params["type"] == "random":
            mask = np.zeros_like(fast_em, dtype=bool)
            mask = mask.flatten()
            mask[
                np.random.choice(
                    mask.shape[0],
                    int(mask.shape[0] * self.params["fraction"]),
                    replace=False,
                )
            ] = 1
            mask = mask.reshape(fast_em.shape)
            return mask, {}
        elif self.params["type"] == "threshold":
            thres = np.quantile(fast_em, self.params["fraction"])
            return fast_em > thres, {}

    def initialize(self):
        pass

    def close(self):
        pass


class GetRescanMapMembraneErrors(GetRescanMap):
    default_params = {
        "em2mb_net": None,
        "error_net": None,
        "device": "auto",
        "pad": 0,
        "rescan_p_thres": 0.1,
        "rescan_ratio": None,
        "search_step": 0.01,
        "do_clahe": False,
    }

    def __init__(self, params=None):
        super().__init__()
        self.params = self.default_params
        if params is not None:
            self.params.update(params)
        assert self.params["em2mb_net"] is not None
        assert self.params["error_net"] is not None
        assert os.path.exists(self.params["em2mb_net"])
        assert os.path.exists(self.params["error_net"])
        assert (self.params["rescan_ratio"] is not None) or (
            self.params["rescan_p_thres"] is not None
        )

    def initialize(self):
        if self.params["device"] == "auto":
            self.params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.params["device"])

        self.em2mb_net = UNet.UNet(1, 2)
        self.em2mb_net.load_state_dict(torch.load(self.params["em2mb_net"]))
        self.em2mb_net.eval()
        self.em2mb_net.to(self.device)

        self.error_net = UNet.UNet(1, 2)
        self.error_net.load_state_dict(torch.load(self.params["error_net"]))
        self.error_net.eval()
        self.error_net.to(self.device)

        trial_data = torch.randn(
            (1, 1, 256, 256), device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            mb = self.em2mb_net(trial_data)
            err = self.error_net(trial_data)

        if self.params["do_clahe"]:
            self.clahe = cv2.createCLAHE(clipLimit=255 * 3.0).apply

    def close(self):
        del self.em2mb_net
        del self.error_net

    def get_rescan_map(self, fast_em):
        if self.params["do_clahe"]:
            fast_em = self.clahe(fast_em)
        mb = tools.get_prob(fast_em, self.em2mb_net)
        error_prob = tools.get_prob(mb, self.error_net, return_dtype=np.float32)

        if self.params["rescan_ratio"] is None:
            rescan_map = self.pad(error_prob > self.params["rescan_p_thres"])
        else:
            warnings.warn("This is very slow in the current implementation.")
            rescan_ratio = self.params["rescan_ratio"]
            imsize = np.prod(error_prob.shape)
            n_target = int(rescan_ratio * imsize)
            thres = np.quantile(error_prob.flatten(), 1 - rescan_ratio)
            rescan_map = self.pad(error_prob > thres)
            while rescan_map.sum() > n_target:
                thres += self.params["search_step"]
                rescan_map = self.pad(error_prob > thres)
        return rescan_map, {"fast_mb": mb, "error_prob": error_prob}

    def pad(self, binim):
        if self.params["pad"] == 0:
            padded = binim
        else:
            padded = skmorph.binary_dilation(
                binim, np.ones((self.params["pad"], self.params["pad"]))
            )
        return padded
