import wandb
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from loguru import logger
import src.training as tr_fn
from datetime import datetime
import pickle
from config import CFG, GCFG

CFG2 = GCFG()

AUTO = tf.data.experimental.AUTOTUNE
SAVE_TIME = datetime.now().strftime("%m%d-%H%M")
class_dict = pickle.load(open("src/class_dict.pkl", "rb"))

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)
replicas = strategy.num_replicas_in_sync

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
    job_type="Learning Rate Finder",
    tags=[CFG2.MODEL, CFG2.OPT, CFG2.LR_SCHED, str(CFG2.IMAGE_SIZE[0])],
    config=CFG,
    config_exclude_keys=[
        "DEBUG", "GCS_REPO", "TRAIN", "ROOT", "DATA", "VERBOSE",
        "DISPLAY_PLOT", "BASE_BATCH_SIZE", "WGTS", "OPT", "LR_SCHED", "MODEL"
    ],
)

logger.info("Building Model...")
with strategy.scope():
    model = tr_fn.create_model(CFG, class_dict)
    opt = tr_fn.create_optimizer(CFG)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=opt, loss=loss, metrics=['sparse_categorical_accuracy'])

logger.info("Establishing learning rate finder")
lrf = tr_fn.LRFinder(model, begin_lr=1e-8, end_lr=1e0, num_epochs=20)
lr_rate = LearningRateScheduler(lrf.lr_schedule)

logger.info("Finding learning rate bounds")
lr_history = lrf.model.fit(
    tr_fn.get_training_dataset(TRAINING_FILENAMES, CFG),
    steps_per_epoch=CFG.STEPS_PER_EPOCH/20,
    validation_data=tr_fn.get_validation_dataset(VALIDATION_FILENAMES, CFG),
    validation_steps=50,
    epochs=CFG.EPOCHS,
    callbacks=[lr_rate],
    verbose=0
)

lrf.lr_plot(lr_history.history['loss'])