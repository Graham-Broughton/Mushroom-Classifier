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
class CFG:
    # GENERAL SETTINGS
    SEED: int = 32
    VERBOSE: int = 2
    REPLICAS: int = 1
    SAVE_TIME: datetime = SAVE_TIME
    DEBUG: bool = False
    FOLDS: int = 5
    DISPLAY_PLOT: bool = True

    # MODEL SETTINGS
    MODEL: str = "swin_large_224"
    MODEL_SIZE: int = 224
    OPT: str = "Adam"
    LR_SCHED: str = "CosineWarmup"
    WGTS: float = 1 / FOLDS
    ES_PATIENCE: int = 5

    # PATHS
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = training
    RAW_DATA: Path = raw_data
    GCS_REPO: str = env.get("GCS_REPO")
    GCS_BASE_MODELS: str = env.get("GCS_BASE_MODELS")
    LOG_FILE: Path = root / "logs" / f"{SAVE_TIME}.log"
    CKPT_DIR: Path = ROOT.parent / "models" / MODEL / SAVE_TIME

    # TFRECORD SETTINGS
    NUM_TRAINING_RECORDS: int = 50
    NUM_VALIDATION_RECORDS: int = 2
    IMAGE_SIZE: List = field(default_factory=lambda: [224, 224])

    # DATASET SETTINGS
    AUGMENT: bool = True
    TTA: int = 11
    EPOCHS: int = 30
    BASE_BATCH_SIZE: int = 4
    BATCH_SIZE: int = 0
    NUM_TRAINING_IMAGES: int = 0
    NUM_VALIDATION_IMAGES: int = 0
    STEPS_PER_EPOCH: int = 0
    VALIDATION_STEPS: int = 0    

    ## LEARNING RATE SETTINGS
    ### Cosine
    # LR_START: float = 0.0001

    ### CosineWarmup
    LR_START: float = 0.00001
    ALPHA: float = 0.00005
    WARMUP_TARGET: float = 0.001

    # ### CosineRestarts
    # LR_START: float = 0.00001
    # ALPHA: float = 0.00005
