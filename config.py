from dataclasses import dataclass
from pathlib import Path
import tensorflow_hub as hub

root = Path.cwd().parent
data = root / "data"
train = data / "train"


@dataclass
class CFG:
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
    MODELS: str = "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224_fe/1"
    IMG_SIZES: int = 224
    FOLDS: int = 5
    BATCH_SIZES: int = 32
    EPOCHS: int = 12
    WGTS = 1 / FOLDS
    # If ViT:
    # PATCH_SIZE: int = map([16] * FOLDS)
    # NUM_PATCHES = (IMG_SIZES // PATCH_SIZE) ** 2

    # Rotational Matrix Settings
    ROT_: float = 180.0
    SHR_: float = 2.0
    HZOOM_: float = 8.0
    WZOOM_: float = 8.0
    HSHIFT_: float = 8.0
    WSHIFT_: float = 8.0

    NUM_CLASSES: int = 489