import math
import os
import pickle
import re

import numpy as np
import tensorflow as tf
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from train_config import CFG

import training.src as src

print(f"Tensorflow version {tf.__version__}")


def batch_to_numpy_images_and_labels(data):
    (images, metas), labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if (
        numpy_labels.dtype == object
    ):  # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return class_dict[label], True
    correct = label == correct_label
    return (
        "{} [{}{}{}]".format(
            class_dict[label],
            "OK" if correct else "NO",
            "\u2192" if not correct else "",
            class_dict[correct_label] if not correct else "",
        ),
        correct,
    )


def display_one_flower(image, title, subplot, red=False, titlesize=16):
    image = (image - image.min()) / (
        image.max() - image.min()
    )  # convert to [0, 1] for avoiding matplotlib warning
    plt.subplot(*subplot)
    plt.axis("off")
    plt.imshow(image)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(titlesize) if not red else int(titlesize / 1.2),
            color="red" if red else "black",
            fontdict={"verticalalignment": "center"},
            pad=int(titlesize / 1.5),
        )
    return (subplot[0], subplot[1], subplot[2] + 1)


def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    (images), (images, predictions), ((images, labels)), ((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(
        zip(images[: rows * cols], labels[: rows * cols])
    ):
        title = "" if label is None else class_dict[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = (
            FIGSIZE * SPACING / max(rows, cols) * 40 + 3
        )  # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(
            image, title, subplot, not correct, titlesize=dynamic_titlesize
        )

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


if __name__ == "__main__":
    AUTO = tf.data.experimental.AUTOTUNE
    CFG = CFG()

    np.random.seed(CFG.SEED)
    tf.random.set_seed(CFG.SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)

    strategy, replicas = src.tpu_test()

    CFG.BATCH_SIZE = CFG.BASE_BATCH_SIZE * replicas

    GCS_PATH_SELECT = {
        192: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-224x224",
        331: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-331x331",
        512: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-512x512",
        None: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-raw",
    }
    GCS_PATH = GCS_PATH_SELECT[None]

    filenames = tf.io.gfile.glob(f"{GCS_PATH}/train*.tfrec")
    filenames, test_filenames = train_test_split(filenames, test_size=1, shuffle=True)
    training_filenames, validation_filenames = train_test_split(
        filenames, test_size=0.15, shuffle=True
    )

    num_train = src.count_data_items(training_filenames)
    num_val = src.count_data_items(validation_filenames)
    num_test = src.count_data_items(test_filenames)

    validation_steps = num_val / CFG.BATCH_SIZE // replicas
    steps_per_epoch = num_train / CFG.BATCH_SIZE // replicas
    TOTAL_STEPS = int(steps_per_epoch * (CFG.EPOCHS - 1))

    class_dict = pickle.load(open("src/class_dict.pkl", "rb"))

    if CFG.DEBUG:
        print("Training data shapes:")
        for (image, meta), label in src.get_batched_dataset(
            training_filenames, CFG, train=True
        ).take(3):
            print(image.numpy().shape, label.numpy().shape)
        print("Training data label examples:", label.numpy())
        print("Validation data shapes:")
        for (image, meta), label in src.get_batched_dataset(
            validation_filenames, CFG
        ).take(3):
            print(image.numpy().shape, label.numpy().shape)
        print("Validation data label examples:", label.numpy())

        # Peek at training data
        training_dataset = src.get_batched_dataset(training_filenames, CFG, train=True)
        training_dataset = training_dataset.unbatch().batch(20)
        train_batch = iter(training_dataset)
        display_batch_of_images(next(train_batch))

    with strategy.scope():
        model = src.create_model()

    config = wandb.helper.parse_config(
        CFG,
        include=(
            "ALPHA",
            "AUGMENT",
            "BATCH_SIZE",
            "EPOCHS",
            "ES_PATIENCE",
            "FOLDS",
            "IMAGE_SIZE",
            "LR_START",
            "MODEL_SIZE",
            "SEED",
            "TTA",
        ),
    )
    wandb.init(
        project="Mushroom-Classifier",
        tags=[CFG.MODEL, CFG.OPT, "ExponentialWarmup", str(CFG.CROP_SIZE[0])],
        config=config,
        dir="../",
    )

    _ = src.get_lr_callback(CFG.BATCH_SIZE, plot=True)

    options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=CFG.ES_PATIENCE,
            verbose=1,
            mode="max",
            restore_best_weights=True,
        ),
        wandb.keras.WandbMetricsLogger(log_freq="batch"),
        wandb.keras.WandbModelCheckpoint(
            str(CFG.CKPT_DIR),  # .h5 for weights, dir for whole model
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            options=options,
            initial_value_threshold=0.84,
        ),
        src.get_lr_callback(CFG.BATCH_SIZE),
    ]

    history = model.fit(
        src.get_batched_dataset(training_filenames, CFG),
        steps_per_epoch=steps_per_epoch,
        epochs=CFG.EPOCHS,
        validation_data=src.get_batched_dataset(validation_filenames, CFG),
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
