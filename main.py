import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import EM_Dataset, download_data, load_volume, select_slices
from src.model import UNet
from src.train import dice_bce_loss, evaluate, train

# Configuration
DATA_DIR = "data"
SAVE_DIR = "models"
NB_TRAIN_LIST = [3, 10, 30]   # number of annotated images used for training
SEEDS = [0, 1, 2, 3, 4]       # each setting is trained once per seed
AUGMENT = True
PATCH_SIZE = 256
NB_PATCHES = 400              # patches per epoch (same for 3 or 30 images)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
NB_TEST_IMAGES = 33           # 33 evenly spaced test slices out of 165, faster than all of them
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(nb_train: int, seed: int, train_images, train_masks, test_images, test_masks) -> float:
    """
    Trains a U-Net on `nb_train` annotated images, evaluates it and saves it.

    Returns:
        float: The Dice score on the test images.
    """
    print(f"\n===== {nb_train} annotated images, seed {seed} ({DEVICE.upper()}) =====")
    np.random.seed(seed)
    torch.manual_seed(seed)

    slices = select_slices(len(train_images), nb_train)
    dataset = EM_Dataset(train_images[slices], train_masks[slices], patch_size=PATCH_SIZE,
                         nb_patches=NB_PATCHES, augment=AUGMENT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # The learning rate goes down to 0 over the epochs, so the last epochs are stable
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        loss, train_dice = train(model, dataloader, dice_bce_loss, optimizer, DEVICE)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss {loss:.4f} | train Dice {train_dice:.4f}")

    dice = evaluate(model, test_images, test_masks, DEVICE)

    # Each seed overwrites the previous model, the app uses the last one
    os.makedirs(SAVE_DIR, exist_ok=True)
    saving_path = os.path.join(SAVE_DIR, f"unet_{nb_train}_images.pth")
    torch.save(model.state_dict(), saving_path)
    print(f"Model saved at: {saving_path}")

    return dice


def run_experiments():
    """
    The main experiment: how does the Dice change with 3, 10 or 30 annotated images?
    """
    download_data(DATA_DIR)

    train_images, train_masks = load_volume(os.path.join(DATA_DIR, "training.tif"),
                                            os.path.join(DATA_DIR, "training_groundtruth.tif"))
    test_images, test_masks = load_volume(os.path.join(DATA_DIR, "testing.tif"),
                                          os.path.join(DATA_DIR, "testing_groundtruth.tif"))
    test_indices = select_slices(len(test_images), NB_TEST_IMAGES)
    test_images, test_masks = test_images[test_indices], test_masks[test_indices]

    results = {}
    for nb_train in NB_TRAIN_LIST:
        results[nb_train] = [run_training(nb_train, seed, train_images, train_masks, test_images, test_masks)
                             for seed in SEEDS]

    print(f"\n===== Results (mean ± std over {len(SEEDS)} seeds, augmentation: {AUGMENT}) =====")
    print("| Annotated images | Test Dice |")
    print("|---|---|")
    for nb_train, scores in results.items():
        print(f"| {nb_train} | {np.mean(scores):.3f} ± {np.std(scores):.3f} |")


if __name__ == "__main__":
    run_experiments()
