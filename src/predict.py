import numpy as np
import torch

from src.dataset import normalize


def predict_mask(image: np.ndarray, model: torch.nn.Module, device: str, threshold: float = 0.5) -> np.ndarray:
    """
    Predicts the mitochondria mask of one full image.

    The model was trained on 256 x 256 patches, but it only contains convolutions,
    so it also works on a full 768 x 1024 image (both divisible by 16).

    Args:
        image (np.ndarray): A grayscale image (H, W), uint8.
        model (torch.nn.Module): The trained model.
        device (str): 'cuda' or 'cpu'.
        threshold (float): Probability above which a pixel is "mitochondria".

    Returns:
        np.ndarray: Binary mask (H, W) with 0.0 / 1.0 values, float32.
    """
    model.eval()

    # (H, W) -> (1, 1, H, W): batch and channel dimensions
    x = torch.from_numpy(normalize(image)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probas = model(x)[0, 0].cpu().numpy()

    return (probas >= threshold).astype(np.float32)
