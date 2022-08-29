import pandas as pd
import os, pickle, yaml, datetime
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import applications, layers
from tensorflow import keras
import tensorflow_addons as tfa
AUTOTUNE = tf.data.experimental.AUTOTUNE

import warnings; warnings.simplefilter('ignore')

from src import functions
from src import NN
from src import datasets

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)

CFG = yaml.safe_load(open('src/config.YAML', 'rb'))
functions.set_seeds(seed=CFG['SEED'])

#save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

train_filenames = tf.io.gfile.glob("gs://mushy_class/tfrecords/train/*FGVC*.tfrec")
val_filenames = tf.io.gfile.glob("gs://mushy_class/tfrecords/val/*FGVC*.tfrec")

STEPS_PER_EPOCH = CFG['NUM_TRAIN_IMAGES'] // CFG['BATCH_SIZE']
VAL_STEPS = CFG['NUM_VAL_IMAGES'] // CFG["BATCH_SIZE"]

def make_callbacks(CFG):

    def lr_scheduler(epoch):
        return CFG['INITIAL_LR_RATE'] * tf.math.pow(CFG['LR_DECAY_FACTOR'], epoch // CFG['EPOCHS_PER_DECAY'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=CFG['ES_PATIENCE'], verbose=1, restore_best_weights=True),

        tf.keras.callbacks.LearningRateScheduler(
            lr_scheduler, verbose=1),
    ]
    return callbacks

with strategy.scope():
    model = NN.create_model(
        lr = CFG['INITIAL_LR_RATE'],
        shape = CFG['IMAGE_SIZE'],
        dropout_pct = CFG['DROPOUT_PCT'],
        classes = CFG['CLASSES']
    )
history = model.fit(datasets.get_dataset(train_filenames, batch_size=CFG['BATCH_SIZE'], DIM=CFG['HEIGHT'], train=True),
                    validation_data=datasets.get_dataset(val_filenames, batch_size=CFG['BATCH_SIZE'], DIM=CFG['HEIGHT'], train=False),
                    validation_steps=VAL_STEPS,
                    epochs=CFG['EPOCHS'],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    callbacks=make_callbacks(CFG))

pickle.dump(history.history, open(f'gs://mushy_class/fgvc_TPU_history-{datetime.datetime.now().strftime("%m%d-%H%M")}.pkl', 'wb'))