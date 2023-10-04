import datetime
import regex
import numpy as np
import tensorflow as tf
import yaml
from src import NN, datasets, functions
import warnings; warnings.simplefilter('ignore')

AUTOTUNE = tf.data.experimental.AUTOTUNE

CFG = yaml.safe_load(open('old_functions/inat_config.YAML', 'rb'))
functions.set_seeds(seed=CFG['SEED'])


def distributed_connection():
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
    return strategy, replicas


def get_filenames(CFG):
    if CFG['TRAINING_SET'] == 'fgvc':
        train_filenames = tf.io.gfile.glob(CFG['FGVC_TRAIN_FILENAMES'])
        val_filenames = tf.io.gfile.glob(CFG['FGVC_VAL_FILENAMES'])
    elif CFG['TRAINING_SET'] == 'inat':
        train_filenames = tf.io.gfile.glob(CFG['INAT_TRAIN_FILENAMES'])
        val_filenames = tf.io.gfile.glob(CFG['INAT_VAL_FILENAMES'])
    elif CFG['TRAINING_SET'] == 'both':
        train_filenames = tf.io.gfile.glob(CFG['TRAIN_FILENAMES'])
        val_filenames = tf.io.gfile.glob(CFG['VAL_FILENAMES'])    
    else:
        print("Training set does not exist")

    return train_filenames, val_filenames


def count_data_items(filenames):
    n = [int(regex.compile(r"-([0-9]*)\.").search(filename).group(1)) 
        for filename in filenames]
    return np.sum(n)


def make_callbacks(CFG):
    log_dir = f"{CFG['LOGGING_PATH']}/fit/{datetime.datetime.now().strftime('%m%d-%H%M')}"

    def lr_scheduler(epoch):
        return CFG['INITIAL_LR_RATE'] * tf.math.pow(CFG['LR_DECAY_FACTOR'], epoch // CFG['EPOCHS_PER_DECAY'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=CFG['ES_PATIENCE'], verbose=1, restore_best_weights=True
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=(50, 250)),
        tf.keras.callbacks.CSVLogger(filename=f'{CFG["LOGGING_PATH"]}/csv_log.csv', separator=',', append=False),
    ]
    return callbacks


def train(CFG, strategy, train_filenames, val_filenames, replicas):

    num_train_images = count_data_items(train_filenames)
    num_val_images = count_data_items(val_filenames)
    STEPS_PER_EPOCH = num_train_images // CFG['BATCH_SIZE'] // replicas
    VAL_STEPS = num_val_images // CFG["BATCH_SIZE"] // replicas
    
    with strategy.scope():
        model = NN.create_model(
            model_path=CFG['MODEL'], lr=CFG['INITIAL_LR_RATE'], shape=CFG['IMAGE_SIZE'], dropout_pct=CFG['DROPOUT_PCT'], classes=CFG['CLASSES']
        )
    history = model.fit(
        datasets.get_dataset(train_filenames, batch_size=CFG['BATCH_SIZE'], DIM=CFG['IMAGE_SIZE'][0], augment=True),
        validation_data=datasets.get_dataset(val_filenames, batch_size=CFG['BATCH_SIZE'], DIM=CFG['IMAGE_SIZE'][0], augment=False),
        validation_steps=VAL_STEPS,
        epochs=CFG['EPOCHS'],
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=make_callbacks(CFG),
    )

    model.save(f'{CFG["GS_BUCKET"]}/models')
    return history


if __name__ == "__main__":
    strategy, replicas = distributed_connection()
    train_filenames, val_filenames = get_filenames(CFG)
    history = train(CFG, strategy, train_filenames, val_filenames, replicas)
    print(history.history)
