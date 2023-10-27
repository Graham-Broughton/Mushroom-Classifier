from dataclasses import dataclass, field
from os import environ as env
from pathlib import Path
from typing import List
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

root = Path.cwd()  # .parent
data = root / "data"
train = data / "train"
SAVE_TIME = datetime.now().strftime("%m%d-%H%M")


@dataclass
class GCFG:
    BATCH_SIZE: int = field(init=False)
    STEPS_PER_EPOCH: int = field(init=False)
    VALIDATION_STEPS: int = field(init=False)
    WGTS: float = field(init=False)

    # GENERAL SETTINGS
    SEED: int = 32
    VERBOSE: int = 2
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = train
    GCS_REPO: str = env.get("GCS_REPO")
    REPLICAS: int = 0
    NUM_TRAINING_IMAGES: int = 0
    NUM_VALIDATION_IMAGES: int = 0
    SAVE_TIME: datetime = SAVE_TIME

    # MODEL SETTINGS
    MODEL: str = "swin_large_384"
    OPT: str = "Adam"
    LR_SCHED: str = "CosineWarmup"

    # TFRECORD SETTINGS
    NUM_TRAINING_RECORDS: int = 107
    NUM_VALIDATION_RECORDS: int = 5
    IMAGE_SIZE: List = field(default_factory=lambda: [384, 384])
    DEBUG: bool = False


@dataclass
class CFG(GCFG):
    # TRAIN SETTINGS
    ## LEARNING RATE SETTINGS
    ### Cosine
    # LR_START: float = 0.0001

    ### CosineWarmup
    LR_START: float = 0.00001
    ALPHA: float = 0.00001
    WARMUP_TARGET: float = 0.002

    ### CosineRestarts
    # LR_START: float = 0.0005

    ### InverseTime

    ## EARLY STOPPING
    ES_PATIENCE: int = 5

    TTAs: int = 11
    DISPLAY_PLOT: bool = True

    ## MODEL SETTINGS
    FOLDS: int = 5
    BASE_BATCH_SIZE: int = 24
    EPOCHS: int = 30

    # DATASET SETTINGS
    AUGMENT: bool = True

    ## Rotational Matrix Settings
    ROT_: float = 180.0
    SHR_: float = 2.0
    HZOOM_: float = 8.0
    WZOOM_: float = 8.0
    HSHIFT_: float = 8.0
    WSHIFT_: float = 8.0

    def __post_init__(self):
        self.BATCH_SIZE = self.BASE_BATCH_SIZE * self.REPLICAS
        self.STEPS_PER_EPOCH = (
            self.NUM_TRAINING_IMAGES // self.BATCH_SIZE // self.REPLICAS
        )
        self.VALIDATION_STEPS = (
            self.NUM_VALIDATION_IMAGES // self.BATCH_SIZE // self.REPLICAS
        )
        self.WGTS = 1 / self.FOLDS
        self.CKPT_DIR: Path = self.ROOT.parent / 'models' / self.MODEL / self.SAVE_TIME
