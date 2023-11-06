import numpy as np
import tensorflow as tf
import os
import re
from prefect import task, Flow


@task
def tpu_test():
    # Detect hardware
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    replicas = strategy.num_replicas_in_sync

    return strategy, replicas


@task
def get_new_cfg(replicas, CFG2, CFG):
    CFG = CFG(
        REPLICAS=replicas,
        NUM_TRAINING_IMAGES=CFG2.NUM_TRAINING_IMAGES,
        NUM_VALIDATION_IMAGES=CFG2.NUM_VALIDATION_IMAGES,
    )
    return CFG


@task
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)


@task
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@task
def select_dataset(CFG2):
    GCS_PATH_SELECT = {
        192: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-224x224v2",
        384: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-384x384",
        512: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-512x512",
    }
    GCS_PATH = GCS_PATH_SELECT[CFG2.IMAGE_SIZE[0]]

    training_filenames = tf.io.gfile.glob(f"{GCS_PATH}/train*.tfrec")
    validation_filenames = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")

    return training_filenames, validation_filenames
