import os, pickle
import tensorflow as tf
from datetime import datetime
import wandb
from wandb.keras import WandbCallback, WandbModelCheckpoint
import numpy as np
from loguru import logger
# from prefect import task, flow
# import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
import src.training as tr_fn
from src.visuals import training_viz
from config import CFG, GCFG


AUTO = tf.data.experimental.AUTOTUNE
class_dict = pickle.load(open("src/class_dict.pkl", "rb"))

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)
replicas = strategy.num_replicas_in_sync


def main(CFG2, CFG, replicas):
    # strategy, replicas = tr_fn.tpu_test()

    logger.info(f"Number of accelerators: {replicas}")

    GCS_PATH_SELECT = {
        192: f"{CFG2.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"{CFG2.GCS_REPO}/tfrecords-jpeg-224x224v2",
        384: f"{CFG2.GCS_REPO}/tfrecords-jpeg-384x384",
        512: f"{CFG2.GCS_REPO}/tfrecords-jpeg-512x512",
    }
    GCS_PATH = GCS_PATH_SELECT[CFG2.IMAGE_SIZE[0]]

    TRAINING_FILENAMES = tf.io.gfile.glob(f"{GCS_PATH}/train*.tfrec")
    VALIDATION_FILENAMES = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")

    CFG2.NUM_TRAINING_IMAGES = tr_fn.count_data_items(TRAINING_FILENAMES)
    CFG2.NUM_VALIDATION_IMAGES = tr_fn.count_data_items(VALIDATION_FILENAMES)

    CFG = CFG(
        REPLICAS=replicas,
        NUM_TRAINING_IMAGES=CFG2.NUM_TRAINING_IMAGES,
        NUM_VALIDATION_IMAGES=CFG2.NUM_VALIDATION_IMAGES,
    )

    wandb.init(
        project="Mushroom-Classifier",
        tags=[CFG2.MODEL, CFG2.OPT, CFG2.LR_SCHED, str(CFG2.IMAGE_SIZE[0])],
        config=CFG,
        dir="../",
        config_exclude_keys=[
            "DEBUG", "GCS_REPO", "TRAIN", "ROOT", "DATA", "VERBOSE", "DISPLAY_PLOT", 
            "BASE_BATCH_SIZE", "WGTS", "OPT", "LR_SCHED", "MODEL"
        ],
    )
    if CFG.DEBUG:
        # data dump
        logger.debug("Training data shapes:")
        for image, label in tr_fn.get_training_dataset(TRAINING_FILENAMES, CFG).take(3):
            logger.debug(f"{image.numpy().shape, label.numpy().shape}")
        logger.debug(f"Training data label examples: {label.numpy()}")
        logger.debug("Validation data shapes:")
        for image, label in tr_fn.get_validation_dataset(VALIDATION_FILENAMES, CFG).take(3):
            logger.debug(f"{image.numpy().shape, label.numpy().shape}")
        logger.debug(f"Validation data label examples: {label.numpy()}")

    logger.info("Building Model...")
    with strategy.scope():
        model = tr_fn.create_model(CFG, class_dict)
        opt = tr_fn.create_optimizer(CFG)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        top3_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=3, name='sparse_top_3_categorical_accuracy'
        )
    model.compile(optimizer=opt, loss=loss, metrics=['sparse_categorical_accuracy', top3_acc])

    logger.info("Training model...")
    # config = wandb.helper.parse_config(CFG, exclude=())
    # wandb.config = config
    history = model.fit(
        tr_fn.get_training_dataset(TRAINING_FILENAMES, CFG),
        steps_per_epoch=CFG.STEPS_PER_EPOCH,
        epochs=CFG.EPOCHS,
        validation_data=tr_fn.get_validation_dataset(VALIDATION_FILENAMES, CFG),
        validation_steps=CFG.VALIDATION_STEPS,
        callbacks=tr_fn.make_callbacks(CFG)
    )

    try:
        os.mkdir(CFG.ROOT / '../models' / CFG.MODEL)
    except FileExistsError:
        pass
    model.save(f'gs://mush-img-repo/models/{CFG.MODEL}/{CFG.SAVE_TIME}')

def get_history(model, fold, files_train, files_valid, CFG):
    logger.info("Training...")
    history = model.fit(
        tr_fn.get_training_dataset(files_train, CFG),
        epochs=CFG.EPOCHS,
        callbacks=tr_fn.make_callbacks(CFG),
        steps_per_epoch=CFG.STEPS_PER_EPOCH,
        validation_data=tr_fn.get_validation_dataset(files_valid, CFG),  # class_weight = {0:1,1:2},
        verbose=CFG.VERBOSE,
    )
    return history


def train(CFG, strategy):
    skf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []
    # preds = np.zeros((count_data_items(files_test), 1))
    GCS_PATH_SELECT = {
        192: f"{CFG2.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"{CFG2.GCS_REPO}/tfrecords-jpeg-224x224v2",
        384: f"{CFG2.GCS_REPO}/tfrecords-jpeg-384x384",
        512: f"{CFG2.GCS_REPO}/tfrecords-jpeg-512x512",
    }
    GCS_PATH = GCS_PATH_SELECT[CFG2.IMAGE_SIZE[0]]   

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(107))):
        # DISPLAY FOLD INFO
        print("#" * 25)
        print("#### FOLD", fold + 1)
        logger.info(
            f"# Image Size {CFG.IMG_SIZES} with Model {CFG.MODELS} and batch_sz {CFG.BATCH_SIZES*CFG.REPLICAS}"
        )

        # CREATE TRAIN AND VALIDATION SUBSETS
        # files_train = tf.io.gfile.glob(
        #     [f"{CFG.GCS_REPO}/train{x:02d}*.tfrec" for x in idxT]
        # )
        # np.random.shuffle(files_train)
        # print("#" * 25)
        # files_valid = tf.io.gfile.glob(
        #     [f"{CFG.GCS_REPO}/train{x:02d}*.tfrec" for x in idxV]
        # )
        # files_test = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")
        files_train = tf.io.gfile.glob(f"{GCS_PATH}/train*.tfrec")
        files_valid = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")

        CFG2.NUM_TRAINING_IMAGES = tr_fn.count_data_items(files_train)
        CFG2.NUM_VALIDATION_IMAGES = tr_fn.count_data_items(files_valid)         

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = tr_fn.create_model(CFG, class_dict)
            opt = tr_fn.create_optimizer(CFG)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()

            top3_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=3, name='sparse_top_3_categorical_accuracy'
            )

        # TRAIN
        history = get_history(model, fold, files_train, files_valid, CFG)

        # PREDICT OOF USING TTA
        logger.info("Predicting OOF with TTA...")
        ds_valid = tr_fn.get_validation_dataset(files_valid, CFG),
        ct_valid = tr_fn.count_data_items(files_valid)
        STEPS = CFG.TTA * ct_valid / CFG.BATCH_SIZES / 4 / CFG.REPLICAS
        pred = model.predict(ds_valid, steps=STEPS, verbose=CFG.VERBOSE)[
            : CFG.TTA * ct_valid,
        ]
        oof_pred.append(np.mean(pred.reshape((ct_valid, CFG.TTA), order="F"), axis=1))
        # oof_pred.append(model.predict(get_dataset(files_valid,dim=CFG.IMG_SIZES),verbose=1))

        # GET OOF TARGETS AND NAMES
        ds_valid = get_dataset(
            files_valid,
            CFG,
            augment=False,
            repeat=False,
            dim=CFG.IMG_SIZES,
            labeled=True,
            return_image_names=True,
        )
        oof_tar.append(
            np.array([target.numpy() for img, target in iter(ds_valid.unbatch())])
        )
        oof_folds.append(np.ones_like(oof_tar[-1], dtype="int8") * fold)
        ds = get_dataset(
            files_valid,
            CFG,
            augment=False,
            repeat=False,
            dim=CFG.IMG_SIZES,
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
            plot_training(history, fold, CFG)


if __name__ == "__main__":
    CFG2 = GCFG()

    print(f"Tensorflow version {tf.__version__}")
    np.set_printoptions(threshold=15, linewidth=80)

    main(CFG2, CFG, replicas)