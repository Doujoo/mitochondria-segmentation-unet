import os
import urllib.request
from typing import Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


# EPFL Hippocampus dataset (Lucchi et al., 2012)
# 165 training slices and 165 test slices of 768 x 1024 pixels
BASE_URL = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus"
FILES = ["training.tif", "training_groundtruth.tif", "testing.tif", "testing_groundtruth.tif"]


def download_data(data_dir: str = "data"):
    """
    Downloads the 4 .tif files of the dataset (about 500 MB) if they are not already there.

    Args:
        data_dir (str): The folder where the files are saved.
    """
    os.makedirs(data_dir, exist_ok=True)
    for file_name in FILES:
        path = os.path.join(data_dir, file_name)
        if not os.path.exists(path):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(f"{BASE_URL}/{file_name}", path)


def load_volume(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a stack of images and its stack of masks.

    Args:
        image_path (str): Path to the .tif file of the images.
        mask_path (str): Path to the .tif file of the annotations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: images (N, H, W) uint8 between 0 and 255,
                                       masks (N, H, W) float32 with 1.0 = mitochondria, 0.0 = background.
    """
    images = tifffile.imread(image_path)
    masks = tifffile.imread(mask_path)

    # Masks are stored as 0 / 255
    masks = (masks > 0).astype(np.float32)
    return images, masks


def select_slices(nb_total: int, nb_wanted: int) -> list[int]:
    """
    Chooses `nb_wanted` slice indices, evenly spaced among `nb_total`.

    Two consecutive slices of the volume are almost identical, so taking
    slices 0, 1, 2 would be like annotating the same image 3 times.
    Example: select_slices(165, 3) -> [0, 55, 110]

    Args:
        nb_total (int): Number of slices available.
        nb_wanted (int): Number of slices to keep.

    Returns:
        list[int]: The chosen indices.
    """
    step = nb_total / nb_wanted
    return [int(i * step) for i in range(nb_wanted)]


class EM_Dataset(Dataset):
    """
    Random patches cut from a few annotated electron-microscopy images.
    """

    def __init__(self, images: np.ndarray, masks: np.ndarray, patch_size: int = 256,
                 nb_patches: int = 400, augment: bool = True):
        """
        Args:
            images (np.ndarray): The annotated images (N, H, W).
            masks (np.ndarray): The corresponding masks (N, H, W).
            patch_size (int): Size of the square patches given to the model.
            nb_patches (int): Number of patches in one epoch. It does not depend on N, so a
                              model trained on 3 images does as many steps as one trained on 30.
            augment (bool): If True, apply random flips and rotations.
        """
        super().__init__()
        self.images = images
        self.masks = masks
        self.patch_size = patch_size
        self.nb_patches = nb_patches
        self.augment = augment

    def __len__(self) -> int:
        return self.nb_patches

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a random patch and its mask. `index` is not used, every call is random.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: image and mask, both (1, patch_size, patch_size) float32.
        """
        i = np.random.randint(0, len(self.images))
        image, mask = self.random_crop(self.images[i], self.masks[i])
        if self.augment:
            image, mask = self.random_flip_rotate(image, mask)
        image = normalize(image)

        # rot90 and fliplr return views with negative strides, torch needs a contiguous copy
        x = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return x, y

    def random_crop(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cuts the same random square of size patch_size in the image and in the mask.
        """
        assert image.shape == mask.shape
        h, w = image.shape
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)

        image = image[top:top + self.patch_size, left:left + self.patch_size]
        mask = mask[top:top + self.patch_size, left:left + self.patch_size]
        return image, mask

    def random_flip_rotate(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random rotation of 0, 90, 180 or 270 degrees, then random horizontal flip.
        The image and the mask get the same transformation.
        """
        k = np.random.randint(0, 4)
        flip = np.random.rand() < 0.5

        image, mask = np.rot90(image, k), np.rot90(mask, k)
        if flip:
            image, mask = np.fliplr(image), np.fliplr(mask)
        return image, mask


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Converts a uint8 image (0-255) to float32 between 0.0 and 1.0.
    Used for training and for prediction, so the model always gets the same kind of input.
    """
    return image.astype(np.float32) / 255.0
