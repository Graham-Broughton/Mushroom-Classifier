from dataclasses import dataclass, field
from pathlib import Path
from os import environ as env
from dotenv import load_dotenv
from typing import List

load_dotenv()

root = Path.cwd().parent
data = root / "data"
train = data / "train"


@dataclass
class CFG:
    # BATCH_SIZE: int
    # NUM_TRAINING_IMAGES: int
    # NUM_VALIDATION_IMAGES: int
    # STEPS_PER_EPOCH: int
    # VALIDATION_STEPS: int
    
    # General Settings
    SEED: int = 42
    VERBOSE: int = 2
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = train
    GCS_REPO: str = env.get("GCS_REPO")

    # Train Settings
    INITIAL_LR_RATE: float    = 0.01
    LR_DECAY_FACTOR: float    = 0.96
    EPOCHS_PER_DECAY: int     = 3
    ES_PATIENCE: int          = 5
    IMAGE_SIZE: List          = field(default_factory=lambda: [224, 224])
    DROPOUT_PCT: float        = 0.1
    BASE_BATCH_SIZE: int      = 32
    EPOCHS: int               = 20
    TTAs: int                 = 11
    DISPLAY_PLOT: bool        = True
    MODEL: str                = 'swin_large_224'

    # Model Settings
    # FOLDS: int = 5
    # WGTS = 1 / FOLDS
    # If ViT:
    # PATCH_SIZE: int = map([16] * FOLDS)
    # NUM_PATCHES = (IMG_SIZES // PATCH_SIZE) ** 2

    # Rotational Matrix Settings
    # ROT_: float = 180.0
    # SHR_: float = 2.0
    # HZOOM_: float = 8.0
    # WZOOM_: float = 8.0
    # HSHIFT_: float = 8.0
    # WSHIFT_: float = 8.0
