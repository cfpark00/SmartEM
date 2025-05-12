import pytest
from autoscript_sdb_microscope_client import SdbMicroscopeClient
import autoscript_sdb_microscope_client.sdb_microscope_client as sdb_microscope_client
import numpy as np


def test_rescan():
    microscope = SdbMicroscopeClient()
    microscope.connect("localhost")
    print(microscope.detector.type.available_values)
