import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import Model, Sequential, layers

from training.src.models.swintransformer import SwinTransformer


def make_callbacks(CFG):
    options = tf.saved_model.SaveOptions(
        experimental_io_device="/job:localhost"
    )  # for whole model saving
    # options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")  # for weights only saving
    CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CFG.ES_PATIENCE,
            verbose=1,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=f"gs://{CFG.GCS_REPO}/logs/{CFG.SAVE_TIME}-csv_log.csv",
            separator=",",
            append=False,
        ),
        wandb.keras.WandbMetricsLogger(log_freq="epoch"),
        wandb.keras.WandbModelCheckpoint(
            str(CFG.CKPT_DIR),  # .h5 for weights, dir for whole model
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            options=options,
            initial_value_threshold=0.8,
        ),
    ]
    return callbacks


def create_model(CFG, class_dict):
    img_adjust_layer = layers.Lambda(
        lambda data: tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(data, tf.float32), mode="torch"
        ),
        input_shape=[*CFG.CROP_SIZE, 3],
    )
    pretrained_model = SwinTransformer(
        CFG.MODEL,
        num_classes=len(class_dict),
        include_top=False,
        pretrained=True,
        use_tpu=True,
    )

    model1 = Sequential(
        [
            img_adjust_layer,
            pretrained_model,
        ]
    )

    # Define the first layer of model2 separately to specify input shape
    input_meta = layers.Input(shape=(6,))
    x = layers.Dense(64, activation="relu")(input_meta)
    x = layers.Dropout(0.5)(x)  # Optional
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    output_meta = layers.Dense(1536)(x)
    model2 = Model(inputs=input_meta, outputs=output_meta)

    # Assuming input_image is the input layer for model1
    input_image = layers.Input(shape=model1.input_shape[1:])
    model1_output = model1(input_image)
    model2_output = model2(input_meta)

    # Concatenate the outputs
    combined = tf.keras.layers.concatenate([model1_output, model2_output])
    combined = layers.BatchNormalization()(combined)
    output = layers.Dense(len(class_dict), activation="softmax")(combined)

    # Create the final model
    final_model = Model(inputs=[input_image, input_meta], outputs=output)

    top3 = tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy")

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", top3],
    )
    final_model.summary()

    return final_model


def get_lr_callback(CFG, batch_size=8, plot=False):
    lr_start = 0.0000005
    lr_max = 0.00000050 * batch_size
    lr_min = 0.0000001
    lr_ramp_ep = 4
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        elif CFG.SCHEDULER == "exp":
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min

        elif CFG.SCHEDULER == "cosine":
            decay_total_epochs = CFG.EPOCHS - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            cosine_decay = 0.4 * (1 + math.cos(phase))
            lr = (lr_max - lr_min) * cosine_decay + lr_min
        return lr

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(CFG.EPOCHS),
            [lrfn(epoch) for epoch in np.arange(CFG.EPOCHS)],
            marker="o",
        )
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title("Learning Rate Scheduler")
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


def create_optimizer(CFG):
    if CFG.LR_SCHED == "CosineWarmup":
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
            CFG.LR_START,
            CFG.STEPS_PER_EPOCH * (CFG.EPOCHS - 2),
            warmup_target=CFG.WARMUP_TARGET,
            warmup_steps=CFG.STEPS_PER_EPOCH * 2,
            alpha=CFG.ALPHA,
            name="CosineWarmup",
        )
    elif CFG.LR_SCHED == "Cosine":
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
            CFG.LR_START,
            CFG.STEPS_PER_EPOCH * CFG.EPOCHS,
            alpha=CFG.ALPHA,
            name="Cosine",
            verbose=1,
        )
    elif CFG.LR_SCHED == "CosineRestarts":
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            CFG.LR_START,
            first_decay_steps=CFG.STEPS_PER_EPOCH * 2,
            alpha=CFG.ALPHA,
        )
    elif CFG.LR_SCHED == "InverseTime":
        learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
            CFG.LR_START, CFG.NUM_TRAINING_IMAGES, decay_rate
        )
    elif CFG.LR_SCHED == "ExpoCustom":
        lr_start = 0.000005
        lr_max = 0.00000125 * CFG.REPLICAS * CFG.BATCH_SIZE
        lr_min = 0.000001
        lr_ramp_ep = 5
        lr_sus_ep = 0
        lr_decay = 0.8
        learning_rate_callback = get_lr_callback(
            lr_start,
            lr_max,
            lr_min,
            lr_ramp_ep,
            lr_sus_ep,
            lr_decay,
            batch_size=CFG.BATCH_SIZE,
        )
    else:
        return tf.keras.optimizers.Adam(0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
    return optimizer
