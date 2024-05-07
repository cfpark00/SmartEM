import pytest
import torch
import numpy as np
from PIL import Image
import os
import tempfile
import shutil


from smartem.offline.train_mb_error_detector.NNtools.Dataset import Dataset


# temp_dir = tempfile.mkdtemp()
# Dataset(temp_dir, subfol=False)
def create_dummy_image(path, size=(1024, 1024)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = Image.fromarray(np.random.randint(0, 256, size, dtype=np.uint8))
    image.save(path)


@pytest.fixture
def setup_dataset(tmp_path):
    # Create a temporary directory
    # temp_dir = tempfile.mkdtemp()
    temp_dir = tmp_path
    frames_dir = os.path.join(tmp_path, "frames")
    masks_dir = os.path.join(tmp_path, "masks")

    # Create subdirectories and images
    subfolders = ["sub1", "sub2"]
    for sub in subfolders:
        os.makedirs(os.path.join(frames_dir, sub))
        os.makedirs(os.path.join(masks_dir, sub))
        for i in range(3):
            img_path = os.path.join(frames_dir, sub, f"img{i}.png")
            mask_path = os.path.join(masks_dir, sub, f"img{i}.png")
            create_dummy_image(img_path)
            create_dummy_image(mask_path)

    yield temp_dir, frames_dir, masks_dir, subfolders

    # Cleanup after test
    shutil.rmtree(temp_dir)


def test_dataset_length(setup_dataset):
    temp_dir, frames_dir, masks_dir, subfolders = setup_dataset
    dataset = Dataset(temp_dir, subfol=True)
    assert len(dataset) == 6, "Dataset length does not match expected number"


def test_dataset_getitem(setup_dataset):
    temp_dir, frames_dir, masks_dir, subfolders = setup_dataset
    dataset = Dataset(temp_dir, subfol=True)
    frame, mask = dataset[0]
    assert isinstance(frame, torch.Tensor), "Frame is not a tensor"
    assert isinstance(mask, torch.Tensor), "Mask is not a tensor"
    assert frame.shape[0] == 1, "Frame should have one channel"
    assert mask.dtype == torch.int64, "Mask tensor should be of type torch.int64"


def test_dataset_get_file_path(setup_dataset):
    temp_dir, frames_dir, masks_dir, subfolders = setup_dataset
    dataset = Dataset(temp_dir, subfol=True)
    frame_path, mask_path = dataset.get_file_path(0)
    assert os.path.exists(frame_path), "Frame path does not exist"
    assert os.path.exists(mask_path), "Mask path does not exist"
