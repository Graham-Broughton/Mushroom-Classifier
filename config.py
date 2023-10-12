from dataclasses import dataclass, field
from pathlib import Path
from os import environ as env
from dotenv import load_dotenv
from typing import List

load_dotenv()

root = Path.cwd()#.parent
data = root / "data"
train = data / "train"


@dataclass
class CFG:
    ## BATCH_SIZE: int
    ## NUM_TRAINING_IMAGES: int
    ## NUM_VALIDATION_IMAGES: int
    ## STEPS_PER_EPOCH: int
    ## VALIDATION_STEPS: int
    # TFRECORD SETTINGS
    NUM_TRAINING_IMAGES: int = 107
    NUM_VALIDATION_IMAGES: int = 5

    # General Settings
    SEED: int = 42
    VERBOSE: int = 2
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = train
    GCS_REPO: str = env.get("GCS_REPO")

    # TRAIN SETTINGS
    ## LEARNING RATE SETTINGS
    LR_START: float = 0.000001
    LR_MAX_BASE: float = 0.0001
    LR_MIN: float = 0.0001
    LR_RAMP_EPOCHS: int = 5
    LR_SUSPEND_EPOCHS: int = 0
    LR_DECAY_FACTOR: float = 0.8
    ES_PATIENCE: int = 5
    IMAGE_SIZE: List = field(default_factory=lambda: [224, 224])
    DROPOUT_PCT: float = 0.1
    BASE_BATCH_SIZE: int = 32
    EPOCHS: int = 20
    TTAs: int = 11
    DISPLAY_PLOT: bool = True
    MODEL: str = "swin_large_224"

    # Model Settings
    # FOLDS: int = 5
    # WGTS = 1 / FOLDS

    # Rotational Matrix Settings
    # ROT_: float = 180.0
    # SHR_: float = 2.0
    # HZOOM_: float = 8.0
    # WZOOM_: float = 8.0
    # HSHIFT_: float = 8.0
    # WSHIFT_: float = 8.0
