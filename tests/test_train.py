import numpy as np
import torch

from src.model import UNet
from src.predict import predict_mask
from src.train import dice_bce_loss, dice_score


def test_dice_identical_masks():
    mask = torch.zeros(64, 64)
    mask[10:30, 10:30] = 1.0
    assert abs(dice_score(mask, mask).item() - 1.0) < 1e-6


def test_dice_disjoint_masks():
    a = torch.zeros(64, 64)
    b = torch.zeros(64, 64)
    a[:10, :10] = 1.0
    b[50:, 50:] = 1.0
    assert dice_score(a, b).item() < 0.01


def test_dice_half_overlap():
    a = torch.zeros(10, 10)
    b = torch.zeros(10, 10)
    a[:, :6] = 1.0   # 60 pixels
    b[:, 3:] = 1.0   # 70 pixels, 30 in common -> Dice = 60 / 130
    assert abs(dice_score(a, b).item() - 60 / 130) < 0.01


def test_loss_is_lower_for_good_prediction():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1.0
    good = target * 0.9 + 0.05
    bad = 1 - good
    assert dice_bce_loss(good, target) < dice_bce_loss(bad, target)


def test_predict_mask():
    model = UNet()
    image = np.random.randint(0, 256, size=(64, 96), dtype=np.uint8)
    mask = predict_mask(image, model, "cpu")
    assert mask.shape == (64, 96)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
