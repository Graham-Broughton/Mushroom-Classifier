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
    SAVE_TIME: datetime = SAVE_TIME
    DEBUG: bool = False
    FOLDS: int = 5
    DISPLAY_PLOT: bool = True
    FIRST_FOLD_ONLY: bool = False

    # MODEL SETTINGS
    MODEL: str = "swinv2_large_256"
    OPT: str = "AdamW"
    LR_SCHED: str = "CosineWarmup"
    WGTS: float = 1 / FOLDS
    ES_PATIENCE: int = 5
    ES_MONITOR: str = "val_accuracy"

    # PATHS
    ROOT: Path = root
    DATA: Path = data
    TRAIN: Path = training
    RAW_DATA: Path = raw_data
    GCS_REPO: str = env.get("GCS_REPO")
    GCS_BASE_MODELS: str = env.get("GCS_BASE_MODELS")
    LOG_FILE: Path = root / "logs" / f"{SAVE_TIME}.log"
    CKPT_DIR: Path = ROOT / "models" / MODEL / SAVE_TIME

    # TFRECORD SETTINGS
    NUM_TRAINING_RECORDS: int = 100
    GCS_IMAGE_SIZE: List = field(default_factory=lambda: [None, None])

    # DATASET SETTINGS
    AUGMENT: bool = True
    TTA: int = 11
    EPOCHS: int = 40
    BASE_BATCH_SIZE: int = 32
    BATCH_SIZE: int = 0
    RAW_SIZE: List = field(default_factory=lambda: [384, 384])
    CROP_SIZE: List = field(default_factory=lambda: [256, 256])

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

    def get_warmup_steps(self, steps_multiplier=1.0):
        self.WARMUP_STEPS = int(self.TRAIN_STEPS * steps_multiplier)
        return self.WARMUP_STEPS
    
    def get_steps(self, num_train_images, num_val_images, batch_size):
        self.TRAIN_STEPS = num_train_images // batch_size
        self.VALIDATION_STEPS = num_val_images // batch_size
        return self.TRAIN_STEPS, self.VALIDATION_STEPS
    
    def get_batch_size(self, replicas):
        self.BATCH_SIZE = self.BASE_BATCH_SIZE * replicas
        return self.BASE_BATCH_SIZE * replicas
