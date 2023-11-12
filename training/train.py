import os
import pickle
import sys

import numpy as np
import src as tr_fn
import tensorflow as tf
import tensorflow.keras.backend as K
import wandb
from loguru import logger
from sklearn.metrics import roc_auc_score

# import mlflow
from sklearn.model_selection import KFold
from train_config import CFG, GCFG

from prefect import Flow


# @Flow
def main(CFG, CFG2, replicas, strategy):
    train_filenames, val_filenames = tr_fn.select_dataset.fn(CFG2)

    # CFG = tr_fn.get_new_cfg(replicas, CFG, train_filenames, val_filenames)
    CFG = CFG(
        REPLICAS=replicas,
        NUM_TRAINING_IMAGES=tr_fn.count_data_items.fn(train_filenames),
        NUM_VALIDATION_IMAGES=tr_fn.count_data_items.fn(val_filenames),
    )

    wandb.init(
        project="Mushroom-Classifier",
        tags=[CFG.MODEL, CFG.OPT, CFG.LR_SCHED, str(CFG.IMAGE_SIZE[0])],
        config=CFG,
        dir="../",
        config_exclude_keys=[
            "DEBUG",
            "GCS_REPO",
            "TRAIN",
            "ROOT",
            "DATA",
            "VERBOSE",
            "DISPLAY_PLOT",
            "BASE_BATCH_SIZE",
            "WGTS",
            "OPT",
            "LR_SCHED",
            "MODEL",
        ],
    )
    if CFG.DEBUG:
        # data dump
        logger.debug("Training data shapes:")
        for image, label in tr_fn.get_training_dataset(train_filenames, CFG).take(3):
            logger.debug(f"{image.numpy().shape, label.numpy().shape}")
        logger.debug(f"Training data label examples: {label.numpy()}")

        logger.debug("Validation data shapes:")
        for image, label in tr_fn.get_validation_dataset(val_filenames, CFG).take(3):
            logger.debug(f"{image.numpy().shape, label.numpy().shape}")
        logger.debug(f"Validation data label examples: {label.numpy()}")

        # Peek at training data
        training_dataset = tr_fn.get_training_dataset()
        training_dataset = training_dataset.unbatch().batch(20)
        train_batch = iter(training_dataset)
        tr_fn.display_batch_of_images(next(train_batch))

    logger.info("Building Model...")
    with strategy.scope():
        model = tr_fn.create_model.fn(CFG, class_dict)

    logger.info("Training model...")
    model.fit(
        tr_fn.get_training_dataset(train_filenames, CFG),
        steps_per_epoch=CFG.STEPS_PER_EPOCH,
        epochs=CFG.EPOCHS,
        validation_data=tr_fn.get_validation_dataset(val_filenames, CFG),
        validation_steps=CFG.VALIDATION_STEPS,
        callbacks=tr_fn.make_callbacks.fn(CFG),
    )

    try:
        os.mkdir(CFG.ROOT / "../models" / CFG.MODEL)
    except FileExistsError:
        pass
    model.save(f"gs://{CFG.GCS_REPO}/models/{CFG.MODEL}/{CFG.SAVE_TIME}")


def get_history(model, files_train, files_valid, CFG):
    logger.info("Training...")
    history = model.fit(
        tr_fn.get_training_dataset(files_train, CFG),
        epochs=CFG.EPOCHS,
        callbacks=tr_fn.make_callbacks.fn(CFG),
        steps_per_epoch=CFG.STEPS_PER_EPOCH,
        validation_data=tr_fn.get_validation_dataset(
            files_valid, CFG
        ),  # class_weight = {0:1,1:2},
        verbose=CFG.VERBOSE,
    )
    return history, model


def train_one_fold(CFG, strategy, fold, files_train, files_valid, *args):
    # BUILD MODEL

    
    return model, oof_pred, oof_tar, oof_val, oof_names, oof_folds


def get_oof_target_names(files_valid, oof_tar, oof_folds, oof_names, CFG, fold):
    # GET OOF TARGETS AND NAMES
    ds_valid = tr_fn.get_dataset(
        files_valid,
        CFG,
        augment=False,
        repeat=False,
        dim=CFG.IMAGE_SIZE,
        labeled=True,
        return_image_names=True,
    )
    oof_tar.append(
        np.array([target.numpy() for img, target in iter(ds_valid.unbatch())])
    )
    oof_folds.append(np.ones_like(oof_tar[-1], dtype="int8") * fold)
    ds = tr_fn.get_dataset(
        files_valid,
        CFG,
        augment=False,
        repeat=False,
        dim=CFG.IMAGE_SIZE,
        labeled=False,
        return_image_names=True,
    )
    oof_names.append(
        np.array(
            [
                img_name.numpy().decode("utf-8")
                for img, img_name in iter(ds.unbatch())
            ]
        )
    )
    return oof_tar, oof_folds, oof_names


def train(CFG, CFG2, replicas, strategy):
    skf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []
    # preds = np.zeros((count_data_items(files_test), 1))
    GCS_PATH_SELECT = {
        192: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-224x224v2",
        256: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-256x256",
        384: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-384x384",
        512: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-512x512",
    }
    GCS_RECORDS_CONVERTER = {
        192: 25,
        224: 33,
        256: 50,
        384: 75,
        512: 100,
    }
    GCS_PATH = GCS_PATH_SELECT[CFG2.IMAGE_SIZE[0]]

    for fold, (idxT, idxV) in enumerate(
        skf.split(np.arange(GCS_RECORDS_CONVERTER[CFG2.IMAGE_SIZE[0]]))
    ):
        # DISPLAY FOLD INFO
        print("#" * 25)
        print("#### FOLD", fold + 1)

        # CREATE TRAIN AND VALIDATION SUBSETS
        # print("#" * 25)
        # files_test = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")
        files_train = tf.io.gfile.glob(
            [f"{GCS_PATH}/train{x:02d}*.tfrec" for x in idxT]
        )
        files_valid = tf.io.gfile.glob(
            [f"{GCS_PATH}/train{x:02d}*.tfrec" for x in idxV]
        )

        CFG = CFG(
            REPLICAS=replicas,
            NUM_TRAINING_IMAGES=tr_fn.count_data_items.fn(files_train),
            NUM_VALIDATION_IMAGES=tr_fn.count_data_items.fn(files_valid),
        )

        logger.info(
            f"# Image Size {CFG.IMAGE_SIZE} with Model {CFG.MODEL} and batch_sz {CFG.BATCH_SIZE}"
        )
        config=wandb.helper.parse_config(
            CFG, include=('ALPHA', 'AUGMENT', 'BATCH_SIZE', 'EPOCHS', 'ES_PATIENCE', 'FOLDS', 'IMAGE_SIZE', 'LR_START', 'MODEL_SIZE', 'SEED', 'TTA')
        )
        wandb.init(
            project="Mushroom-Classifier", tags=[CFG.MODEL, CFG.OPT, CFG.LR_SCHED, str(CFG.IMAGE_SIZE[0]), str(fold)],
            config=config, dir="../", group='cross_val',
            config_exclude_keys=[
                "DEBUG",
                "GCS_REPO",
                "TRAIN",
                "ROOT",
                "DATA",
                "RAW_DATA",
                "VERBOSE",
                "DISPLAY_PLOT",
                "BASE_BATCH_SIZE",
                "WGTS",
                "OPT",
                "LR_SCHED",
                "MODEL",
            ],)
        
        K.clear_session()
        with strategy.scope():
            model = tr_fn.create_model.fn(CFG, class_dict)

        # TRAIN
        logger.info("Training...")
        history = model.fit(
            tr_fn.get_training_dataset(files_train, CFG),
            epochs=CFG.EPOCHS,
            callbacks=tr_fn.make_callbacks.fn(CFG),
            steps_per_epoch=CFG.STEPS_PER_EPOCH,
            validation_data=tr_fn.get_validation_dataset(
                files_valid, CFG
            ),  # class_weight = {0:1,1:2},
            verbose=CFG.VERBOSE,
        )

        # PREDICT OOF USING TTA
        logger.info("Predicting OOF with TTA...")
        ds_valid = (tr_fn.get_validation_dataset(files_valid, CFG),)
        ct_valid = tr_fn.count_data_items.fn(files_valid)
        STEPS = CFG.TTA * ct_valid / CFG.BATCH_SIZE / 4 / CFG.REPLICAS
        pred = model.predict(ds_valid, steps=STEPS, verbose=CFG.VERBOSE)[
            : CFG.TTA * ct_valid,
        ]
        oof_pred.append(np.mean(pred.reshape((ct_valid, CFG.TTA), order="F"), axis=1))
        # oof_pred.append(model.predict(get_dataset(files_valid,dim=CFG.IMAGE_SIZE),verbose=1))

        # GET OOF TARGETS AND NAMES
        ds_valid = tr_fn.get_dataset(
            files_valid,
            CFG,
            augment=False,
            repeat=False,
            dim=CFG.IMAGE_SIZE,
            labeled=True,
            return_image_names=True,
        )
        oof_tar.append(
            np.array([target.numpy() for img, target in iter(ds_valid.unbatch())])
        )
        oof_folds.append(np.ones_like(oof_tar[-1], dtype="int8") * fold)
        ds = tr_fn.get_dataset(
            files_valid,
            CFG,
            augment=False,
            repeat=False,
            dim=CFG.IMAGE_SIZE,
            labeled=False,
            return_image_names=True,
        )
        oof_names.append(
            np.array(
                [
                    img_name.numpy().decode("utf-8")
                    for img, img_name in iter(ds.unbatch())
                ]
            )
        )

        # REPORT RESULTS
        auc = roc_auc_score(oof_tar[-1], oof_pred[-1])
        oof_val.append(np.max(history.history["val_auc"]))
        logger.info(
            f"#### FOLD {fold + 1} OOF AUC without TTA = {oof_val[-1]}, with TTA = {auc}"
        )

        # PLOT TRAINING
        if CFG.DISPLAY_PLOT:
            tr_fn.plot_training(history, fold, CFG)
        
        wandb.finish()


if __name__ == "__main__":
    CFG2 = GCFG()

    logger.remove(0)
    logger.add(sink=f"{CFG.ROOT}/logs/" + "log_{time}.log", level="WARNING")
    logger.add(sink=sys.stdout, level="DEBUG", colorize=True)

    AUTO = tf.data.experimental.AUTOTUNE

    class_dict = pickle.load(open("class_dict.pkl", "rb"))

    strategy, replicas = tr_fn.tpu_test.fn()
    CFG2.REPLICAS = replicas

    logger.info(f"Number of accelerators: {replicas}")
    logger.debug(f"Tensorflow version {tf.__version__}")

    # main(CFG2, CFG, replicas, strategy)
    main(CFG, CFG2, replicas, strategy)
