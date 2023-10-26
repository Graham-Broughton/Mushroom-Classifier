import tensorflow as tf
from src.models.swintransformer import SwinTransformer
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
        # wandb.keras.WandbMetricsLogger(log_freq='batch'),
        # wandb.keras.WandbModelCheckpoint(
        #     str(CFG.ROOT / '../models' / CFG.MODEL / f"{save_time}.h5"),  # .h5 for weights, dir for whole model
        #     monitor='val_loss', verbose=1, save_best_only=True,
        #     save_weights_only=True, options=options,
        # )
    ]
    return callbacks


def create_model(CFG, class_dict):
    img_adjust_layer = tf.keras.layers.Lambda(
        lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), 
        input_shape=[*CFG.IMAGE_SIZE, 3]
    )
    model = SwinTransformer(CFG.MODEL, num_classes=len(class_dict), include_top=False, pretrained=False, use_tpu=True)
    model = tf.keras.Sequential([
        img_adjust_layer,
        model,
        tf.keras.layers.Dense(len(class_dict), activation='softmax')
    ])
    model.load_weights(CFG.ROOT / 'base_models' / CFG.MODEL / 'base_model.h5')
    return model


def create_optimizer(CFG):
    learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
        CFG.LR_START,
        CFG.NUM_TRAINING_IMAGES * CFG.EPOCHS,
        alpha=CFG.ALPHA,
        name="Cosine_Schedular",
    )
    optimizer = tf.keras.optimizers.Adam(0.0001)
    # optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
    return optimizer