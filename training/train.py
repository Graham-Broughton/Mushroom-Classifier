import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from pickle import load
import numpy as np
import wandb
from loguru import logger
from train_config import CFG
import warnings
import os
import src
from sklearn.model_selection import KFold
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

CFG = CFG()


# @Flow
def main(CFG, replicas, strategy):
    train_filenames, val_filenames = src.select_dataset.fn(CFG)

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
        for image, label in src.get_training_dataset(train_filenames, CFG).take(3):
            logger.debug(f"{image.numpy().shape, label.numpy().shape}")
        logger.debug(f"Training data label examples: {label.numpy()}")

        logger.debug("Validation data shapes:")
        for image, label in src.get_validation_dataset(val_filenames, CFG).take(3):
            logger.debug(f"{image.numpy().shape, label.numpy().shape}")
        logger.debug(f"Validation data label examples: {label.numpy()}")

        # Peek at training data
        training_dataset = src.get_training_dataset()
        training_dataset = training_dataset.unbatch().batch(20)
        train_batch = iter(training_dataset)
        src.display_batch_of_images(next(train_batch))

    logger.info("Building Model...")
    with strategy.scope():
        model = src.create_model.fn(CFG, class_dict)

    logger.info("Training model...")
    model.fit(
        src.get_training_dataset(train_filenames, CFG),
        steps_per_epoch=CFG.STEPS_PER_EPOCH,
        epochs=CFG.EPOCHS,
        validation_data=src.get_validation_dataset(val_filenames, CFG),
        validation_steps=CFG.VALIDATION_STEPS,
        callbacks=src.make_callbacks.fn(CFG),
    )

    try:
        os.mkdir(CFG.ROOT / "../models" / CFG.MODEL)
    except FileExistsError:
        pass
    model.save(f"gs://{CFG.GCS_REPO}/models/{CFG.MODEL}/{CFG.SAVE_TIME}")


def get_history(model, files_train, files_valid, CFG):
    logger.info("Training...")
    history = model.fit(
        src.get_training_dataset(files_train, CFG),
        epochs=CFG.EPOCHS,
        callbacks=src.make_callbacks.fn(CFG),
        steps_per_epoch=CFG.STEPS_PER_EPOCH,
        validation_data=src.get_validation_dataset(
            files_valid, CFG
        ),  # class_weight = {0:1,1:2},
        verbose=CFG.VERBOSE,
    )
    return history, model

def get_oof_target_names(files_valid, oof_tar, oof_folds, oof_names, CFG, fold):
    # GET OOF TARGETS AND NAMES
    ds_valid = src.get_dataset(
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
    ds = src.get_dataset(
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


def train(CFG, replicas, strategy):
    skf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []
    # preds = np.zeros((count_data_items(files_test), 1))

    GCS_PATH, NUM_RECORDS = src.path_records_converter.fn(CFG.IMAGE_SIZE[0], CFG)

    for fold, (idxT, idxV) in enumerate(
        skf.split(np.arange(NUM_RECORDS))
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


        logger.info(
            f"# Image Size {CFG.IMAGE_SIZE} with Model {CFG.MODEL} and batch_sz {CFG.BATCH_SIZE}"
        )
        config=wandb.helper.parse_config(
            CFG, include=('ALPHA', 'AUGMENT', 'BATCH_SIZE', 'EPOCHS', 'ES_PATIENCE', 'FOLDS', 'IMAGE_SIZE', 'LR_START', 'MODEL_SIZE', 'SEED', 'TTA')
        )
        wandb.init(
            project="Mushroom-Classifier", tags=[CFG.MODEL, CFG.OPT, CFG.LR_SCHED, str(CFG.IMAGE_SIZE[0]), str(fold)],
            config=config, dir="../", group='cross_val',
            )
        
        K.clear_session()
        with strategy.scope():
            model = src.create_model.fn(CFG, class_dict)

        # TRAIN
        logger.info("Training...")
        history = model.fit(
            src.get_training_dataset(files_train, CFG),
            epochs=CFG.EPOCHS,
            callbacks=src.make_callbacks.fn(CFG),
            steps_per_epoch=CFG.STEPS_PER_EPOCH,
            validation_data=src.get_validation_dataset(
                files_valid, CFG
            ),  # class_weight = {0:1,1:2},
            verbose=CFG.VERBOSE,
        )

        # PREDICT OOF USING TTA
        logger.info("Predicting OOF with TTA...")
        ds_valid = (src.get_validation_dataset(files_valid, CFG),)
        ct_valid = src.count_data_items.fn(files_valid)
        STEPS = CFG.TTA * ct_valid / CFG.BATCH_SIZE / 4 / CFG.REPLICAS
        pred = model.predict(ds_valid, steps=STEPS, verbose=CFG.VERBOSE)[
            : CFG.TTA * ct_valid,
        ]
        oof_pred.append(np.mean(pred.reshape((ct_valid, CFG.TTA), order="F"), axis=1))
        # oof_pred.append(model.predict(get_dataset(files_valid,dim=CFG.IMAGE_SIZE),verbose=1))

        # GET OOF TARGETS AND NAMES
        ds_valid = src.get_dataset(
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
        ds = src.get_dataset(
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
            src.plot_training(history, fold, CFG)
        
        wandb.finish()


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sink=f"{CFG.ROOT}/logs/" + "log_{time}.log", level="WARNING")
    logger.add(sink=sys.stdout, level="DEBUG", colorize=True)

    AUTO = tf.data.experimental.AUTOTUNE

    class_dict = load(open("class_dict.pkl", "rb"))

    strategy, replicas = src.tpu_test.fn()
    CFG.REPLICAS = replicas

    logger.info(f"Number of accelerators: {replicas}")
    logger.debug(f"Tensorflow version {tf.__version__}")

    # main(CFG2, CFG, replicas, strategy)
    main(CFG, replicas, strategy)
