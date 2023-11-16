import tensorflow as tf
from sklearn.model_selection import train_test_split
from pickle import load
import wandb
from loguru import logger
from train_config import CFG
import warnings
from src.training.dataset2 import get_training_dataset, get_batched_dataset
from src.training.NN2 import create_model, get_callbacks
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

CFG = CFG()
AUTO = tf.data.AUTOTUNE
class_dict = load(open("class_dict.pkl", "rb"))

if __name__ == "__main__":
    try:  # detect TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')  # TPU detection
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:  # detect GPUs
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU

    REPLICAS = strategy.num_replicas_in_sync
    logger.debug(f"Number of Accelerators: {strategy.num_replicas_in_sync}")

    CFG.BATCH_SIZE =  REPLICAS * CFG.BASE_BATCH_SIZE

    training_filenames, validation_filenames, test_filenames, num_train, num_val = get_training_dataset(CFG)

    CFG.TOTAL_STEPS = int((num_train / CFG.BATCH_SIZE) * CFG.EPOCHS - 1)
    validation_steps = num_val // CFG.BATCH_SIZE
    steps_per_epoch = num_train // CFG.BATCH_SIZE

    CFG.WARMUP_STEPS = steps_per_epoch * 2
    CFG.DECAY_STEPS = steps_per_epoch * CFG.EPOCHS - CFG.WARMUP_STEPS

    logger.info("Initializing Model...")
    with strategy.scope():
        model = create_model(CFG)
        callbacks = get_callbacks(CFG)
    
    config=wandb.helper.parse_config(
        CFG, include=(
            'ALPHA', 'AUGMENT', 'BATCH_SIZE', 'EPOCHS', 'ES_PATIENCE', 'IMAGE_SIZE', 'DECAY_STEPS',
            'LR_START', 'MODEL_SIZE', 'SEED', 'TTA', 'TOTAL_STEPS', 'WARMUP_STEPS', 'WARMUP_TARGET',
            'RAW_SIZE',
        )
    )
    wandb.init(
        project="Mushroom-Classifier",
        tags=[CFG.MODEL, CFG.OPT, CFG.LR_SCHED, str(CFG.IMAGE_SIZE[0])],
        config=config,
        dir="../",
    )

    tr_ds = get_batched_dataset(training_filenames, CFG, train=True)
    val_ds = get_batched_dataset(validation_filenames, CFG)

    history = model.fit(
        tr_ds, 
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=CFG.EPOCHS, 
        callbacks=callbacks
        )