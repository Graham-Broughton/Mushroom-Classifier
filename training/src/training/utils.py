import numpy as np
from loguru import logger
import tensorflow as tf
import os
import regex as re


def tpu_test(CFG2, CFG):
    # Detect hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:  # If TPU not found
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    replicas = strategy.num_replicas_in_sync
    logger.info("Number of accelerators: ", replicas)

    CFG = get_new_cfg(replicas, CFG2, CFG)

    return strategy, CFG


def get_new_cfg(replicas, CFG2, CFG):
    CFG = CFG(
        REPLICAS=replicas,
        NUM_TRAINING_IMAGES=CFG2.NUM_TRAINING_IMAGES,
        NUM_VALIDATION_IMAGES=CFG2.NUM_VALIDATION_IMAGES,
    )
    return CFG


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)