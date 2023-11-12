from dataclasses import dataclass, field
from datetime import datetime
from os import environ as env
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

root = Path(env.get("PYTHONPATH").split(":")[0])
training = root / "training"
data = training / "data"
raw_data = data / "raw"
SAVE_TIME = datetime.now().strftime("%m%d-%H%M")


@dataclass
class GCFG:
    BATCH_SIZE: int = 0
    STEPS_PER_EPOCH: int = 0
    VALIDATION_STEPS: int = 0
    WGTS: float = 0

    # GENERAL SETTINGS
    SEED: int = 42
    VERBOSE: int = 2
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = training
    RAW_DATA: Path = raw_data
    GCS_REPO: str = env.get("GCS_REPO")
    GCS_BASE_MODELS: str = env.get("GCS_BASE_MODELS")
    REPLICAS: int = 0
    NUM_TRAINING_IMAGES: int = 0
    NUM_VALIDATION_IMAGES: int = 0
    SAVE_TIME: datetime = SAVE_TIME
    LOG_FILE: Path = root / "logs" / f"{SAVE_TIME}.log"
    FOLDS: int = 5

    # MODEL SETTINGS
    MODEL: str = "swin_large_224"
    MODEL_SIZE: int = 224
    OPT: str = "Adam"
    LR_SCHED: str = "CosineRestarts"
    BASE_BATCH_SIZE: int = 16

    # TFRECORD SETTINGS
    NUM_TRAINING_RECORDS: int = 50
    NUM_VALIDATION_RECORDS: int = 2
    IMAGE_SIZE: List = field(default_factory=lambda: [224, 224])
    DEBUG: bool = True

    # DATASET SETTINGS


@dataclass
class CFG(GCFG):
    # TRAIN SETTINGS
    ## LEARNING RATE SETTINGS
    ### Cosine
    # LR_START: float = 0.0001

    ### CosineWarmup
    # LR_START: float = 0.00001
    # ALPHA: float = 0.00001
    # WARMUP_TARGET: float = 0.002

    ### CosineRestarts
    LR_START: float = 0.0008
    ALPHA: float = 0.01

    ### InverseTime

    ## EARLY STOPPING
    ES_PATIENCE: int = 5

    TTA: int = 11
    DISPLAY_PLOT: bool = True

    ## MODEL SETTINGS
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
            self.NUM_TRAINING_IMAGES / self.BATCH_SIZE // self.REPLICAS
        )
        self.VALIDATION_STEPS = (
            self.NUM_VALIDATION_IMAGES / self.BATCH_SIZE // self.REPLICAS
        )
        self.WGTS = 1 / self.FOLDS
        self.CKPT_DIR: Path = self.ROOT.parent / "models" / self.MODEL / self.SAVE_TIME
