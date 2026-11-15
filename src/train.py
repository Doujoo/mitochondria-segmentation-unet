from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.predict import predict_mask


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Dice = 2 * |pred AND target| / (|pred| + |target|)
    1.0 = perfect overlap, 0.0 = no overlap at all.

    Works with binary masks (0/1) and with probabilities (soft Dice, used in the loss).

    Args:
        pred (torch.Tensor): Predicted mask or probabilities, any shape.
        target (torch.Tensor): True mask (0/1), same shape.

    Returns:
        torch.Tensor: The Dice score (a single number).
    """
    smooth = 1.0  # avoids dividing by 0, and gives 1.0 when both masks are empty
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def dice_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss = BCE + (1 - Dice).
    Mitochondria are only ~6% of the pixels, so a model that predicts background
    everywhere already has a small BCE. The Dice part punishes that.
    """
    return F.binary_cross_entropy(pred, target) + (1 - dice_score(pred, target))


def train(model: torch.nn.Module, dataloader: DataLoader, loss_fn: Callable,
          optimizer: torch.optim.Optimizer, device: str) -> Tuple[float, float]:
    """
    Trains the model for one epoch.

    Returns:
        Tuple[float, float]: The average loss and the average Dice (threshold 0.5) over the epoch.
    """
    model.to(device)
    model.train()
    total_loss = 0
    total_dice = 0

    loop = tqdm(dataloader, desc="Training")
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        dice = dice_score((pred > 0.5).float(), y).item()
        total_loss += loss.item()
        total_dice += dice
        loop.set_postfix(loss=loss.item(), dice=dice)

    return total_loss / len(dataloader), total_dice / len(dataloader)


def evaluate(model: torch.nn.Module, images: np.ndarray, masks: np.ndarray, device: str) -> float:
    """
    Evaluates the model on full test images (not on patches).

    Args:
        model (torch.nn.Module): The trained model.
        images (np.ndarray): Test images (N, H, W) uint8.
        masks (np.ndarray): True masks (N, H, W) with 0/1 values.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: The average Dice over the test images.
    """
    model.to(device)
    scores = []
    for image, mask in tqdm(zip(images, masks), total=len(images), desc="Evaluation"):
        pred = predict_mask(image, model, device)
        scores.append(dice_score(torch.from_numpy(pred), torch.from_numpy(mask)).item())

    mean_dice = float(np.mean(scores))
    print(f"Test Dice: {mean_dice:.4f}")
    return mean_dice
