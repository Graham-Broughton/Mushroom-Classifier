import pickle
from datetime import datetime
from loguru import logger

import numpy as np
import tensorflow as tf
import wandb
from src.training import utils, dataset
from src.visuals import training_viz
from wandb.keras import WandbCallback, WandbModelCheckpoint
# from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
# from prefect import task, flow
from config import CFG, GCFG

utils.check_for_colab()


CFG2 = GCFG()

AUTO = tf.data.experimental.AUTOTUNE
print(f"Tensorflow version {tf.__version__}")
np.set_printoptions(threshold=15, linewidth=80)

SAVE_TIME = datetime.now().strftime("%m%d-%H%M")
LOG_DIR = f"{CFG2.GCS_REPO}/logs/{CFG2.MODEL}/{SAVE_TIME}"

class_dict = pickle.load(open("src/class_dict.pkl", "rb"))

wandb.init(
    project="Mushroom-Classifier",
    tags=[f"{CFG2.MODEL}, {CFG2.OPT}, {CFG2.LR_SCHED}, {str(CFG2.IMAGE_SIZE[0])}"],
)


utils.tpu_test(CFG2)

GCS_PATH_SELECT = {
    192: f"{CFG2.GCS_REPO}/tfrecords-jpeg-192x192",
    224: f"{CFG2.GCS_REPO}/tfrecords-jpeg-224x224v2",
    331: f"{CFG2.GCS_REPO}/tfrecords-jpeg-331x331",
    512: f"{CFG2.GCS_REPO}/tfrecords-jpeg-512x512",
}
GCS_PATH = GCS_PATH_SELECT[CFG2.IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(f"{GCS_PATH}/train*.tfrec")
VALIDATION_FILENAMES = tf.io.gfile.glob(f"{GCS_PATH}/val*.tfrec")


CFG2.NUM_TRAINING_IMAGES = utils.count_data_items(TRAINING_FILENAMES)
CFG2.NUM_VALIDATION_IMAGES = utils.count_data_items(VALIDATION_FILENAMES)

CFG = CFG(
    REPLICAS=CFG2.REPLICAS,
    NUM_TRAINING_IMAGES=CFG2.NUM_TRAINING_IMAGES,
    NUM_VALIDATION_IMAGES=CFG2.NUM_VALIDATION_IMAGES,
)


# data dump
logger.debug("Training data shapes:")
for image, label in dataset.get_training_dataset().take(3):
    logger.debug(image.numpy().shape, label.numpy().shape)
logger.debug("Training data label examples:", label.numpy())
logger.debug("Validation data shapes:")
for image, label in dataset.get_validation_dataset().take(3):
    logger.debug(image.numpy().shape, label.numpy().shape)
logger.debug("Validation data label examples:", label.numpy())


if CFG.DEBUG:
    # Peek at training data
    training_dataset = dataset.get_training_dataset()
    training_dataset = training_dataset.unbatch().batch(20)
    train_batch = iter(training_dataset)
    training_viz.display_batch_of_images(next(train_batch))





def build_model(model, num_classes, dim=128):
    inp = tf.keras.layers.Input(shape=(dim, dim, 3))
    base = hub.KerasLayer(
        model, trainable=True
    )  # (input_shape=(dim,dim,3),weights='imagenet',include_top=False)
    x = base(inp, training=True)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def sv(fold):
    return tf.keras.callbacks.ModelCheckpoint(
        f"{GCS_PATH}/models/-{CFG.IMG_SIZES}fold-{fold}.h5",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )


def get_lr_callback(batch_size=8):
    lr_start = 0.000005
    lr_max = 0.00000125 * CFG.REPLICAS * batch_size
    lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


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
    strategy, tpu = tpu_test(CFG)
    train(CFG, strategy)
