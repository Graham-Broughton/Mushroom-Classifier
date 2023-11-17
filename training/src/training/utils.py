import numpy as np
import tensorflow as tf
import os
import re
import random


def tpu_test():
    # Detect hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    except ValueError: 
        tpu = None

    if tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy() # Default strategy that works on CPU and single GPU
        print('Running on CPU instead')
    replicas = strategy.num_replicas_in_sync
    return strategy, replicas

def get_new_cfg(replicas, CFG, train_filenames, val_filenames):
    CFG = CFG(
        REPLICAS=replicas,
        NUM_TRAINING_IMAGES=count_data_items.fn(train_filenames),
        NUM_VALIDATION_IMAGES=count_data_items.fn(val_filenames),
    )
    return CFG

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 

def select_dataset(CFG2):
    GCS_PATH_SELECT = {
        192: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-224x224v2",
        256: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-256x256",
        384: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-384x384",
        512: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-512x512",
        None: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-raw",
    }
    GCS_PATH = GCS_PATH_SELECT[CFG2.IMAGE_SIZE[0]]

    training_filenames = tf.io.gfile.glob(f"{GCS_PATH}/train*.tfrec")
    validation_filenames = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")

    return training_filenames, validation_filenames
