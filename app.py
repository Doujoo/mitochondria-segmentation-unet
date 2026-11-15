import os

import numpy as np
import streamlit as st
import tifffile
import torch
from PIL import Image

from src.model import UNet
from src.predict import predict_mask
from src.train import dice_score

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "models"
EXAMPLES_DIR = "examples"
NB_TRAIN_LIST = [3, 10, 30]
# Slices of the test volume, never seen during training
EXAMPLES = {
    "Test slice 20": "test_slice_20",
    "Test slice 100": "test_slice_100",
    "Test slice 140": "test_slice_140",
}


@st.cache_resource(show_spinner=False)
def load_model(nb_train: int) -> torch.nn.Module:
    """
    Loads the U-Net trained on `nb_train` images (saved by main.py).
    """
    model = UNet()
    path = os.path.join(MODELS_DIR, f"unet_{nb_train}_images.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def read_image(uploaded_file) -> np.ndarray:
    """
    Reads an uploaded file (.tif or .png/.jpg) as a grayscale uint8 image (H, W).
    """
    if uploaded_file.name.lower().endswith((".tif", ".tiff")):
        image = tifffile.imread(uploaded_file)
        if image.ndim == 3:  # a stack of slices: keep the first one
            image = image[0]
    else:
        image = np.array(Image.open(uploaded_file).convert("L"))
    return image.astype(np.uint8)


def load_example(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads an example image and its true mask (0/1) from EXAMPLES_DIR.
    """
    image = np.array(Image.open(os.path.join(EXAMPLES_DIR, f"{name}.png")).convert("L"))
    mask = np.array(Image.open(os.path.join(EXAMPLES_DIR, f"{name}_mask.png")))
    return image, (mask > 0).astype(np.float32)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Colors the predicted mitochondria in magenta on top of the grayscale image.

    Returns:
        np.ndarray: RGB image (H, W, 3) uint8.
    """
    rgb = np.stack([image, image, image], axis=-1).astype(np.float32)
    magenta = np.array([230, 70, 160], dtype=np.float32)
    rgb[mask == 1] = 0.5 * rgb[mask == 1] + 0.5 * magenta
    return rgb.astype(np.uint8)


def main():
    """
    Streamlit page: upload an image and compare the predictions of the 3 models.
    """
    st.set_page_config(page_title="EM Segmentation", page_icon="🔬", layout="wide")
    st.title("Mitochondria segmentation with few annotations")
    st.write("Compare U-Nets trained on 3, 10 and 30 annotated electron-microscopy images.")

    example = st.selectbox("Example image (from the test set)", list(EXAMPLES))
    uploaded_file = st.file_uploader("Or upload your own electron-microscopy image",
                                     type=["tif", "tiff", "png", "jpg"])
    threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)

    # The ground truth is only known for the examples
    if uploaded_file is not None:
        image, truth = read_image(uploaded_file), None
    else:
        image, truth = load_example(EXAMPLES[example])

    # The U-Net needs a height and width divisible by 16
    h, w = image.shape
    image = image[: h - h % 16, : w - w % 16]

    panels = [(image, "Input image")]
    if truth is not None:
        truth = truth[: h - h % 16, : w - w % 16]
        panels.append((overlay(image, truth), "Ground truth"))

    columns = st.columns(len(panels) + len(NB_TRAIN_LIST))
    for column, (panel, caption) in zip(columns, panels):
        column.image(panel, caption=caption, width="stretch")

    for column, nb_train in zip(columns[len(panels):], NB_TRAIN_LIST):
        model = load_model(nb_train)
        mask = predict_mask(image, model, DEVICE, threshold)
        caption = f"{nb_train} annotated images"
        if truth is not None:
            dice = dice_score(torch.from_numpy(mask), torch.from_numpy(truth)).item()
            caption += f" (Dice {dice:.3f})"
        column.image(overlay(image, mask), caption=caption, width="stretch")


if __name__ == "__main__":
    main()
