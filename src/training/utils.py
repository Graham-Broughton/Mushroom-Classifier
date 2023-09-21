import numpy as np
from loguru import logger
import tensorflow as tf
from dotenv import load_dotenv, set_key
import os

load_dotenv()


def tpu_test(CFG):
    DEVICE = os.environ['DEVICE']
    if DEVICE == "TPU":
        logger.info("connecting to TPU...")
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            logger.info('Running on TPU ', tpu.master())
        except ValueError:
            logger.info("Could not connect to TPU")
            tpu = None

        if tpu:
            try:
                logger.info("initializing  TPU ...")
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                logger.info("TPU initialized")
            except _:
                logger.info("failed to initialize TPU")
        else:
            set_key(CFG['ROOT'], 'DEVICE', "GPU")

    if os.environ['DEVICE'] != "TPU":
        logger.info("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()

    if os.environ['DEVICE'] == "GPU":
        logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    CFG.REPLICAS = strategy.num_replicas_in_sync
    return strategy, tpu


def count_data_items(filenames):
    return len(filenames)
