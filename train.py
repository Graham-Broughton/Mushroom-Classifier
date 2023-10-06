import pickle
from datetime import datetime

import tensorflow as tf
import wandb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from config import CFG
from src.models.swintransformer import SwinTransformer
from src.training.dataset import (
    count_data_items,
    get_training_dataset,
    get_validation_dataset,
)
from src.visuals.training_viz import display_batch_of_images, display_training_curves

# CFG = CFG()

print(f"Tensorflow version {tf.__version__}")
AUTO = tf.data.experimental.AUTOTUNE


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
print(f"Number of accelerators: {replicas}")

CFG.BATCH_SIZE = CFG.BASE_BATCH_SIZE * replicas
GCS_DS_SELECT = {
    192: f"{CFG.GCS_REPO}/tfrecords-jpeg-192x192",
    224: f"{CFG.GCS_REPO}/tfrecords-jpeg-224x224",
    331: f"{CFG.GCS_REPO}/tfrecords-jpeg-331x331",
    512: f"{CFG.GCS_REPO}/tfrecords-jpeg-512x512",
}
print(CFG)
CFG = CFG()
print(CFG)
GCS_DS = GCS_DS_SELECT[CFG.IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(f"{GCS_DS}/train*.tfrec")
VALIDATION_FILENAMES = tf.io.gfile.glob(f"{GCS_DS}/val*.tfrec")

class_dict = pickle.load(open("src/class_dict.pkl", "rb"))

print("Training data shapes:")
for image, label in get_training_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())
print("Validation data shapes:")
for image, label in get_validation_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Validation data label examples:", label.numpy())


CFG.NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
CFG.NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
CFG.STEPS_PER_EPOCH = CFG.NUM_TRAINING_IMAGES // CFG.BATCH_SIZE // replicas
CFG.VALIDATION_STEPS = CFG.NUM_VALIDATION_IMAGES // CFG.BATCH_SIZE // replicas


def get_lr_callback():
    lr_start = 0.000005
    lr_max = 0.00000125 * replicas * CFG.BATCH_SIZE
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


def make_callbacks(CFG):
    log_dir = f"{CFG.GCS_REPO}/logs/{datetime.now().strftime('%m%d-%H%M')}"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=CFG["ES_PATIENCE"],
            verbose=1,
            restore_best_weights=True,
        ),
        get_lr_callback(),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=(50, 250)),
        tf.keras.callbacks.CSVLogger(
            filename=f'{CFG.GCS_REPO}/logs/{datetime.now().strftime("%m%d-%H%M")}-csv_log.csv',
            separator=",",
            append=False,
        ),
    ]
    return callbacks


with strategy.scope():
    img_adjust_layer = tf.keras.layers.Lambda(
        lambda data: tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(data, tf.float32), mode="torch"
        ),
        input_shape=[*CFG.IMAGE_SIZE, 3],
    )
    pretrained_model = SwinTransformer(
        CFG.MODEL,
        num_classes=len(class_dict),
        include_top=False,
        pretrained=True,
        use_tpu=True,
    )

    model = tf.keras.Sequential(
        [
            img_adjust_layer,
            pretrained_model,
            tf.keras.layers.Dense(len(class_dict), activation="softmax"),
        ]
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

history = model.fit(
    get_training_dataset(),
    steps_per_epoch=CFG.STEPS_PER_EPOCH,
    epochs=CFG.EPOCHS,
    validation_data=get_validation_dataset(),
    validation_steps=CFG.VALIDATION_STEPS,
    callbacks=make_callbacks(CFG),
)

display_training_curves(
    history.history["loss"], history.history["val_loss"], "loss", 211
)
display_training_curves(
    history.history["sparse_categorical_accuracy"],
    history.history["val_sparse_categorical_accuracy"],
    "accuracy",
    212,
)
