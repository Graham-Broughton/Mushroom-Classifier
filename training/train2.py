from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
from pickle import load
import re
import numpy as np
import wandb
from loguru import logger
from train_config import CFG
import warnings
from tfswin import SwinTransformerV2Large256, preprocess_input
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

CFG = CFG()

try:  # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:  # detect GPUs
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU

REPLICAS = strategy.num_replicas_in_sync
print("Number of Accelerators: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE
class_dict = load(open("class_dict.pkl", "rb"))


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def decode_image(image_data, CFG):
    image = tf.image.decode_jpeg(image_data, channels=3)  # image format uint8 [0,255]
    image = tf.cast(image, tf.uint8)
    image = tf.image.resize_with_crop_or_pad(image, 384, 384)
    image = tf.image.resize(image, size=CFG.IMAGE_SIZE, method="lanczos5")
    
    return image

def read_labeled_tfrecord(example, CFG):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/id": tf.io.FixedLenFeature([], tf.string),
        "image/meta/dataset": tf.io.FixedLenFeature([], tf.int64),
        "image/meta/longitude": tf.io.FixedLenFeature([], tf.float32),
        "image/meta/latitude": tf.io.FixedLenFeature([], tf.float32),
        "image/meta/date": tf.io.FixedLenFeature([], tf.string),
        "image/meta/class_priors": tf.io.FixedLenFeature([], tf.float32),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        "image/class/text": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = decode_image(example["image/encoded"], CFG)
    label = tf.cast(example["image/class/label"], tf.int32)
    return image, label

def load_dataset(filenames, CFG):
  # read from TFRecords. For optimal performance, read from multiple
  # TFRecord files at once and set the option experimental_deterministic = False
  # to allow order-altering optimizations.

  option_no_order = tf.data.Options()
  option_no_order.experimental_deterministic = False

  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
  dataset = dataset.with_options(option_no_order)
  dataset = dataset.map(lambda x: read_labeled_tfrecord(x, CFG), num_parallel_calls=AUTO)
  return dataset

def get_model(model_url: str, res: int = 256, num_classes: int = 467) -> tf.keras.Model:
    inputs = layers.Input(shape=(*res, 3), dtype='int8')
    outputs = SwinTransformerV2Large256(include_top=False, pooling='avg', input_shape=[*res, 3])(inputs)
    outputs = layers.Dense(num_classes, activation='softmax')(outputs)
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def get_batched_dataset(filenames, CFG, train=False):
  dataset = load_dataset(filenames, CFG)
  dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTO)
  # dataset = dataset.cache() # This dataset fits in RAM
  if train:
    # Best practices for Keras:
    # Training dataset: repeat then batch
    # Evaluation dataset: do not repeat
    dataset = dataset.repeat()
    # dataset = dataset.shuffle(BATCH_SIZE * 10)
  dataset = dataset.batch(CFG.BATCH_SIZE, drop_remainder=True)
  dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
  # should shuffle too but this dataset was well shuffled on disk already
  return dataset

def main(CFG2, CFG, replicas):
    # strategy, replicas = tr_fn.tpu_test()

    logger.info(f"Number of accelerators: {replicas}")

    GCS_PATH_SELECT = {
        192: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-224x224v2",
        384: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-384x384",
        512: f"gs://{CFG2.GCS_REPO}/tfrecords-jpeg-512x512",
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
        config_exclude_keys=["DEBUG", "GCS_REPO", "TRAIN", "ROOT", "DATA", "VERBOSE", "DISPLAY_PLOT", "BASE_BATCH_SIZE", "WGTS", "OPT", "LR_SCHED", "MODEL"],
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
    model.save(f'../models/{CFG.MODEL}/{CFG.SAVE_TIME}')

def get_history(model, fold, files_train, files_valid, CFG):
    logger.info("Training...")
    history = model.fit(
        get_dataset(
            files_train,
            CFG,
            augment=True,
            shuffle=True,
            repeat=True,
            dim=CFG.IMG_SIZES,
            batch_size=CFG.BATCH_SIZES,
        ),
        epochs=CFG.EPOCHS,
        callbacks=[sv(fold), get_lr_callback(CFG.BATCH_SIZES)],
        steps_per_epoch=count_data_items(files_train) / CFG.BATCH_SIZES // CFG.REPLICAS,
        validation_data=get_dataset(
            files_valid,
            CFG,
            augment=False,
            shuffle=False,
            repeat=False,
            dim=CFG.IMG_SIZES,
        ),  # class_weight = {0:1,1:2},
        verbose=CFG.VERBOSE,
    )
    return history.history


def train(CFG, strategy):
    skf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []
    # preds = np.zeros((count_data_items(files_test), 1))

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(107))):
        # DISPLAY FOLD INFO
        print("#" * 25)
        print("#### FOLD", fold + 1)
        print(
            f"#### Image Size {CFG.IMG_SIZES} with {CFG.MODELS} and batch_size {CFG.BATCH_SIZES * CFG.REPLICAS}"
        )
        logger.info(
            f"# Image Size {CFG.IMG_SIZES} with Model {CFG.MODELS} and batch_sz {CFG.BATCH_SIZES*CFG.REPLICAS}"
        )

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob(
            [f"{GCS_PATH}/train{x:02d}*.tfrec" for x in idxT]
        )
        np.random.shuffle(files_train)
        print("#" * 25)
        files_valid = tf.io.gfile.glob(
            [f"{GCS_PATH}/train{x:02d}*.tfrec" for x in idxV]
        )
        files_test = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = build_model(CFG.MODELS, CFG.NUM_CLASSES, dim=CFG.IMG_SIZES)

        # TRAIN
        history = get_history(model, fold, files_train, files_valid, CFG)

        logger.info("Loading best model...")
        model.load_weights(f"fold-{fold}.h5")

        # PREDICT OOF USING TTA
        logger.info("Predicting OOF with TTA...")
        ds_valid = get_dataset(
            files_valid,
            CFG,
            labeled=False,
            return_image_names=False,
            augment=True,
            repeat=True,
            shuffle=False,
            dim=CFG.IMG_SIZES,
            batch_size=CFG.BATCH_SIZES * 4,
        )
        ct_valid = count_data_items(files_valid)
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