import numpy as np
import pytest
import torch

from src.dataset import EM_Dataset, normalize, select_slices


@pytest.fixture
def fake_data():
    """3 fake images of 300 x 400 with a square "mitochondria" in each one."""
    images = np.random.randint(0, 256, size=(3, 300, 400), dtype=np.uint8)
    masks = np.zeros((3, 300, 400), dtype=np.float32)
    masks[:, 100:150, 200:260] = 1.0
    return images, masks


def test_select_slices():
    assert select_slices(165, 3) == [0, 55, 110]
    assert len(select_slices(165, 30)) == 30


def test_normalize():
    image = np.array([[0, 255]], dtype=np.uint8)
    result = normalize(image)
    assert result.dtype == np.float32
    assert result.min() == 0.0 and result.max() == 1.0


def test_random_crop_shape(fake_data):
    images, masks = fake_data
    dataset = EM_Dataset(images, masks, patch_size=128)
    image, mask = dataset.random_crop(images[0], masks[0])
    assert image.shape == (128, 128)
    assert mask.shape == (128, 128)


def test_random_flip_rotate_same_for_image_and_mask(fake_data):
    """The image and the mask must be transformed in the same way."""
    images, _ = fake_data
    square = images[0, :128, :128]
    dataset = EM_Dataset(images, images.astype(np.float32), patch_size=128)
    for _ in range(10):
        image, mask = dataset.random_flip_rotate(square, square.astype(np.float32))
        assert np.array_equal(image, mask)


def test_getitem(fake_data):
    images, masks = fake_data
    dataset = EM_Dataset(images, masks, patch_size=128, nb_patches=10)
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (1, 128, 128)
    assert y.shape == (1, 128, 128)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert len(dataset) == 10
