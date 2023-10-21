import numpy as np
from loguru import logger
import tensorflow as tf
# from dotenv import load_dotenv, set_key, find_dotenv
import os
import regex as re


def tpu_test(CFG):
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
    CFG.REPLICAS = strategy.num_replicas_in_sync
    logger.info("Number of accelerators: ", strategy.num_replicas_in_sync)


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def check_for_colab():
    from sys import exit as xt
    from subprocess import run

    print("Checking for colab environment Dependencies")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        from google.colab import auth
        auth.authenticate_user()
        print("Found Colab Environment")
        run('pip install -q tensorflow==2.10.0 wandb python-dotenv tensorboard_plugin_profile tensorflow_io==0.27.0', shell=True)
        try:
            os.remove("./tmp")
            os.chdir("/content/drive/MyDrive/Mushroom-Classifier")
        except FileNotFoundError:
            os.mkdir("./tmp")
            print("Restarting Runtime")
            xt("Restart Runtime")
    except ImportError:
        if os.path.isdir('../Mushroom-Classifier') is True:
            print("Found Desktop/VM Environment")
        else:
            print("Must run from root directory of project")


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
