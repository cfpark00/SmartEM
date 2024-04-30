import pytest
import torch
from skimage import measure
import smartem
import unittest
import numpy as np
from smartem.online.get_rescan_maps import GetRescanMapMembraneErrors
from smartem.online.get_rescan_maps import GetRescanMapTest
from smartem.offline.train_mb_error_detector.NNtools import UNet
import tempfile
import os,sys

class TestGetRescanMapMembraneErrors(unittest.TestCase):


    def setUp(self):
        # Create U-Net models and save their state dicts to temporary files
        self.em2mb_net = UNet.UNet(1, 2)
        self.error_net = UNet.UNet(1, 2)

        # Temporary files to store models
        self.temp_em2mb_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_error_file = tempfile.NamedTemporaryFile(delete=False)

        # Save models
        torch.save(self.em2mb_net.state_dict(), self.temp_em2mb_file.name)
        torch.save(self.error_net.state_dict(), self.temp_error_file.name)

        # Parameters with paths to the temporary model files
        self.params = {
            "em2mb_net": self.temp_em2mb_file.name,
            "error_net": self.temp_error_file.name,
            "device": "auto"  # Set to auto to use GPU if available
        }

    def tearDown(self):
        # Close and remove temporary files
        self.temp_em2mb_file.close()
        os.remove(self.temp_em2mb_file.name)
        self.temp_error_file.close()
        os.remove(self.temp_error_file.name)

    def test_initialize(self):
        # Initialize rescan map object with the parameters
        rescan_map = GetRescanMapMembraneErrors(params=self.params)
        rescan_map.initialize()

        # Assertions to check if models are loaded correctly and device is set
        self.assertIsInstance(rescan_map.em2mb_net, UNet.UNet)
        self.assertIsInstance(rescan_map.error_net, UNet.UNet)
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(expected_device)
        self.assertEqual(str(rescan_map.device), expected_device)

        # Optional: Check CUDA availability directly
        if torch.cuda.is_available():
            self.assertTrue('cuda' in str(rescan_map.device))
        else:
            self.assertEqual(str(rescan_map.device), 'cpu')

        # Optional: Further checks can include verifying the state dicts if necessary
        # This could be done by comparing the loaded model's state dict with the original one
        original_em2mb_state_dict = torch.load(self.temp_em2mb_file.name, map_location=rescan_map.device)
        loaded_em2mb_state_dict = rescan_map.em2mb_net.state_dict()
        self.assertTrue(all(torch.equal(original_em2mb_state_dict[k], loaded_em2mb_state_dict[k]) for k in original_em2mb_state_dict))

        original_error_state_dict = torch.load(self.temp_error_file.name, map_location=rescan_map.device)
        loaded_error_state_dict = rescan_map.error_net.state_dict()
        self.assertTrue(all(torch.equal(original_error_state_dict[k], loaded_error_state_dict[k]) for k in original_error_state_dict))


class TestGetRescanMapTest(unittest.TestCase):


    def test_custom_initialization(self):
        """ Test the initialization with custom parameters. """
        custom_params = {"type": "random", "fraction": 0.3}
        rescan = GetRescanMapTest(params=custom_params)
        self.assertEqual(rescan.params["type"], "random")
        self.assertEqual(rescan.params["fraction"], 0.3)
    
    def test_get_rescan_map_threshold(self):
        """ Test the threshold rescan map type. """
        fast_em = np.array([[0.1, 0.2], [0.9, 0.95]])
        rescan = GetRescanMapTest(params={"type": "threshold", "fraction": 0.5})
        mask, _ = rescan.get_rescan_map(fast_em)
        expected_mask = np.array([[False, False], [True, True]], dtype=bool)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_invalid_type_initialization(self):
        """ Test initialization with an invalid type to ensure it raises an assertion error. """
        with self.assertRaises(AssertionError):
            GetRescanMapTest(params={"type": "invalid_type"})

    def test_get_rescan_map_half(self):
        """ Test the half rescan map type. """
        fast_em = np.random.rand(100, 100)
        rescan = GetRescanMapTest(params={"type": "half"})
        mask, _ = rescan.get_rescan_map(fast_em)
        expected_mask = np.zeros_like(fast_em, dtype=bool)
        expected_mask[:50, :] = 1
        np.testing.assert_array_equal(mask, expected_mask)

    def test_get_rescan_map_random(self):
        """ Test the random rescan map type. """
        fast_em = np.random.rand(10, 10)
        rescan = GetRescanMapTest(params={"type": "random", "fraction": 0.2})
        mask, _ = rescan.get_rescan_map(fast_em)
        self.assertEqual(np.sum(mask), 20)  # 20% of 100 elements is 20


if __name__ == '__main__':
    unittest.main()


        
