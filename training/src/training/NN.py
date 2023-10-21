import tensorflow as tf
from transformers import TFViTForImageClassification, ViTImageProcessor
import wandb


def make_callbacks(CFG, save_time):
    # options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')  # for whole model saving
    options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")  # for weights only saving

    callbacks = [
        tf.keras.callbacks.CSVLogger(
            filename=f'{CFG.GCS_REPO}/logs/{save_time}-csv_log.csv',
            separator=",",
            append=False,
        ),
        wandb.keras.WandbMetricsLogger(log_freq='batch'),
        wandb.keras.WandbModelCheckpoint(
            str(CFG.ROOT / 'models' / CFG.MODEL / f"{save_time}.h5"),  # .h5 for weights, dir for whole model
            monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, options=options,
        )
    ]
    return callbacks


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


def create_model(CFG, class_dict):
    # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    model = tf.keras.Sequential([
        processor
    ])