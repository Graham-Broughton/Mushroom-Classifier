import numpy as np
import tensorflow as tf
import os
import re


def tpu_test():
    # Detect hardware
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    replicas = strategy.num_replicas_in_sync

    return strategy, replicas


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