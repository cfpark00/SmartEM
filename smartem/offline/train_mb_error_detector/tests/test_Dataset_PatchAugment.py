import pytest
import numpy as np
import h5py
import tempfile
import os
import torch
from smartem.offline.train_mb_error_detector.NNtools.Dataset import PatchAugmentDataset

@pytest.fixture
def create_hdf5(tmp_path):
    # Setup a temporary HDF5 file
    # temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file = tmp_path / "temp.h5"
    print("File path:", temp_file)  # Directly printing the file path
    with h5py.File(temp_file, "w") as f:
        size = 1024
        f.attrs['W'] = size
        f.attrs['H'] = size
        f.attrs['dwts'] = [50, 800]
        f.attrs['regs'] = ['region1', 'region2']
        f.create_dataset('region1/50/im', data=np.random.rand(size, size) * 255)
        f.create_dataset('region1/50/mask', data=np.random.randint(0, 1, (size, size)) * 255)
        f.create_dataset('region2/800/im', data=np.random.rand(size, size) * 255)
        f.create_dataset('region2/800/mask', data=np.random.randint(0, 1, (size, size)) * 255)
        print("File contents", print_hdf5_contents(temp_file))
        print()
    yield temp_file
    os.unlink(temp_file)  # Clean up the file after tests are done


def print_hdf5_contents(file_path):
    with h5py.File(file_path, "r") as f:
        print("Attributes:")
        for attr in f.attrs:
            print(f"{attr}: {f.attrs[attr]}")

        def print_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
            else:
                print(f"Group: {name}")

        f.visititems(print_group)

def test_print_hdf5_contents(create_hdf5):
    file_path = create_hdf5
    print_hdf5_contents(file_path)


# def test_get_item(create_hdf5):
#     file_path = create_hdf5
#     dataset = PatchAugmentDataset(file_path, 10, lambda x: x, lambda x: x)
#     image, mask = dataset[0]
    
#     assert isinstance(image, torch.Tensor) and image.dtype == torch.float32
#     assert isinstance(mask, torch.Tensor) and mask.dtype == torch.int64
#     assert image.shape == (1, dataset.out, dataset.out)
#     assert mask.shape == (dataset.out, dataset.out)

# def test_random_shape_gen(create_hdf5):
#     file_path = create_hdf5
#     dataset = PatchAugmentDataset(file_path, 10, lambda x: x, lambda x: x)
#     pad_shape = dataset.random_shape_gen(dataset.grid, 0.5)
    
#     assert isinstance(pad_shape, np.ndarray)
#     assert pad_shape.shape == dataset.grid.shape[1:]  # Should match the shape of the grid sliced
#     assert pad_shape.dtype == bool

# def test_get_random_image_mask(create_hdf5):
#     file_path = create_hdf5
#     dataset = PatchAugmentDataset(file_path, 10, lambda x: x, lambda x: x)
#     image, mask, reg, dwt = dataset.get_random_image_mask(dataset.p_dwts_biased)
    
#     assert isinstance(image, np.ndarray) and image.dtype == float
#     assert isinstance(mask, np.ndarray) and mask.dtype == float
#     assert reg in dataset.regs
#     assert dwt in dataset.dwts

# def test_get_random_image_mask_from_reg(create_hdf5):
#     file_path = create_hdf5
#     dataset = PatchAugmentDataset(file_path, 10, lambda x: x, lambda x: x)
#     reg = 'region1'
#     image, mask, reg_out, dwt = dataset.get_random_image_mask_from_reg(reg, dataset.p_dwts_biased)
    
#     assert isinstance(image, np.ndarray) and image.dtype == float
#     assert isinstance(mask, np.ndarray) and mask.dtype == float
#     assert reg_out == reg
#     assert dwt in dataset.dwts
