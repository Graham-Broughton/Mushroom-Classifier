from dataclasses import dataclass
from pathlib import Path
from typing import List

root = Path.cwd().parent
data = root / "data"
train = data / "train"


@dataclass
class Config:
    # General Settings
    SEED: int = 42
    VERBOSE: int = 2
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = train

    # Train Settings
    TTA: int = 11
    DISPLAY_PLOT: bool = True

    # Model Settings
    MODELS: List[str] = []
    IMG_SIZES: List[int] = [128, 128, 128, 128, 128]  #  Choose number of sizes corresponding to number of folds
    FOLDS: int = 5
    BATCH_SIZE: int = 32 * FOLDS
    EPOCHS: int = 12 * FOLDS
    WGTS = [1 / FOLDS] * FOLDS
    # If ViT:
    PATCH_SIZE: int
    NUM_PATCHES = (IMG_SIZES // PATCH_SIZE) ** 2
