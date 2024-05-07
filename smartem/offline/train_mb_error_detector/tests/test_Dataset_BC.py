import pytest
import torch
import numpy as np
from PIL import Image
import os
import tempfile
import shutil
from torchvision.transforms import ToTensor, Normalize


from smartem.offline.train_mb_error_detector.NNtools.Dataset import Dataset

def create_dummy_image(path, size=(1024, 1024)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = Image.fromarray(np.random.randint(0, 256, size, dtype=np.uint8))
    image.save(path)

@pytest.fixture
def setup_dataset(tmp_path):
    # Create a temporary directory
    temp_dir = tmp_path
    frames_dir = os.path.join(tmp_path, "frames")
    masks_dir = os.path.join(tmp_path, "masks")

    # Create subdirectories and images
    subfolders = ['sub1', 'sub2']
    for sub in subfolders:
        os.makedirs(os.path.join(frames_dir, sub))
        os.makedirs(os.path.join(masks_dir, sub))
        for i in range(3):
            img_path = os.path.join(frames_dir, sub, f'img{i}.png')
            mask_path = os.path.join(masks_dir, sub, f'img{i}.png')
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

    # Iterate over all images in the dataset
    for i in range(len(dataset)):
        frame_tensor, mask_tensor = dataset[i]
        
        # Build the path to the original image and mask
        subfolder_index = i // 3  # adjust based on number of images per folder
        frame_file = os.path.join(frames_dir, subfolders[subfolder_index], f'img{i % 3}.png')
        mask_file = os.path.join(masks_dir, subfolders[subfolder_index], f'img{i % 3}.png')
        
        # Load the original images and normalize
        frame = np.array(Image.open(frame_file)) / 255.0
        mask = np.array(Image.open(mask_file)) / 255.0  # Normalization step if required
        
        # Convert to tensors
        frame_tensor_manual = torch.from_numpy(frame).unsqueeze(0).to(dtype=torch.float32)  # Add channel dimension
        mask_tensor_manual = torch.from_numpy(mask).to(dtype=torch.int64)
        

        # Check types and shapes
        assert isinstance(frame_tensor, torch.Tensor), "Frame is not a tensor"
        assert isinstance(mask_tensor, torch.Tensor), "Mask is not a tensor"
        assert frame_tensor.dtype == torch.float32, "Frame tensor should be of type torch.float32"
        assert mask_tensor.dtype == torch.int64, "Mask tensor should be of type torch.int64"
        assert frame_tensor.shape[0] == 1, "Frame should have one channel"  # Assuming channel first format


        # Compare tensors
        # torch.testing.assert_close(frame_tensor, frame_tensor_manual, rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(mask_tensor, mask_tensor_manual, rtol=1e-5, atol=1e-8)

        



    # frame, mask = dataset[0]
    # assert isinstance(frame, torch.Tensor), "Frame is not a tensor"
    # assert isinstance(mask, torch.Tensor), "Mask is not a tensor"
    # assert frame.shape[0] == 1, "Frame should have one channel"
    # assert mask.dtype == torch.int64, "Mask tensor should be of type torch.int64"

def test_dataset_get_file_path(setup_dataset):
    temp_dir, frames_dir, masks_dir, subfolders = setup_dataset
    dataset = Dataset(temp_dir, subfol=True)
    frame_path, mask_path = dataset.get_file_path(0)
    assert os.path.exists(frame_path), "Frame path does not exist"
    assert os.path.exists(mask_path), "Mask path does not exist"


    # check frame and mask are the same ones as saved in the frames_dir and masks_dir