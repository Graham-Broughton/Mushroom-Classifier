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
class GCFG:
    BATCH_SIZE: int = field(init=False)
    STEPS_PER_EPOCH: int = field(init=False)
    VALIDATION_STEPS: int = field(init=False)
    WGTS: float = field(init=False)

    REPLICAS: int = 0
    NUM_TRAINING_IMAGES: int = 0
    NUM_VALIDATION_IMAGES: int = 0

    # GENERAL SETTINGS
    SEED: int = 42
    VERBOSE: int = 2
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = train
    GCS_REPO: str = env.get("GCS_REPO")
    MODEL: str = "swin_large_224"    

    # TFRECORD SETTINGS
    NUM_TRAINING_RECORDS: int = 107
    NUM_VALIDATION_RECORDS: int = 5
    IMAGE_SIZE: List = field(default_factory=lambda: [224, 224])



@dataclass
class CFG(GCFG):
    # TRAIN SETTINGS
    ## LEARNING RATE SETTINGS
    LR_START: float = 0.0004
    BETA1: float = 0.9
    BETA2: float = 0.99
    DECAY_STEPS: int = 1000
    ALPHA: float = 0.000001

    ## EARLY STOPPING
    ES_PATIENCE: int = 5

    TTAs: int = 11
    DISPLAY_PLOT: bool = True

    ## MODEL SETTINGS
    FOLDS: int = 5
    DROPOUT_PCT: float = 0.1
    BASE_BATCH_SIZE: int = 8
    EPOCHS: int = 20
    
    # OLD LR SCHED
    # LR_START: float = 0.000001
    # LR_MAX_BASE: float = 0.0001
    # LR_MIN: float = 0.0001
    # LR_RAMP_EPOCHS: int = 5
    # LR_SUSPEND_EPOCHS: int = 0
    # LR_DECAY_FACTOR: float = 0.8

    # Rotational Matrix Settings
    # ROT_: float = 180.0
    # SHR_: float = 2.0
    # HZOOM_: float = 8.0
    # WZOOM_: float = 8.0
    # HSHIFT_: float = 8.0
    # WSHIFT_: float = 8.0

    def __post_init__(self):
        self.BATCH_SIZE = self.BASE_BATCH_SIZE * self.REPLICAS
        self.STEPS_PER_EPOCH = self.NUM_TRAINING_IMAGES // self.BATCH_SIZE // self.REPLICAS
        self.VALIDATION_STEPS = self.NUM_VALIDATION_IMAGES // self.BATCH_SIZE // self.REPLICAS
        self.WGTS = 1 / self.FOLDS


