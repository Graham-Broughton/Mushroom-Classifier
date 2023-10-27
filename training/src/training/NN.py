import tensorflow as tf
from src.models.swintransformer import SwinTransformer
import wandb


def make_callbacks(CFG):
    options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')  # for whole model saving
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
            filename=f'{CFG.GCS_REPO}/logs/{CFG.SAVE_TIME}-csv_log.csv',
            separator=",",
            append=False,
        ),
        wandb.keras.WandbMetricsLogger(log_freq='epoch'),
        wandb.keras.WandbModelCheckpoint(
            str(CFG.CKPT_DIR),  # .h5 for weights, dir for whole model
            monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, options=options,
            initial_value_threshold=0.8
        )
    ]
    return callbacks


def create_model(CFG, class_dict):
    img_adjust_layer = tf.keras.layers.Lambda(
        lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), 
        input_shape=[*CFG.IMAGE_SIZE, 3]
    )
    pretrained_model = tf.keras.models.load_model(CFG.ROOT / 'base_models' / CFG.MODEL / 'base_model', compile=False)
    model = tf.keras.Sequential([
        img_adjust_layer,
        pretrained_model,
        tf.keras.layers.Dense(len(class_dict), activation='softmax')
    ])
    return model


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, args):
        lr_start = 0.000005
        lr_max = 0.00000125 * CFG.REPLICAS * batch_size
        lr_min = 0.000001
        lr_ramp_ep = 5
        lr_sus_ep = 0
        lr_decay = 0.8

    def __call__(self, step):
        return self.initial_learning_rate / (step + 1)


def get_lr_callback(*args, batch_size=8):
    def lrfn(epoch):
        if epoch < args.lr_ramp_ep:
            lr = (args.lr_max - args.lr_start) / args.lr_ramp_ep * epoch + args.lr_start

        elif epoch < args.lr_ramp_ep + args.lr_sus_ep:
            lr = args.lr_max

        else:
            lr = (args.lr_max - args.lr_min) * args.lr_decay ** (
                epoch - args.lr_ramp_ep - args.lr_sus_ep
            ) + args.lr_min

        return lr
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
            1000
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
        learning_rate_callback = get_lr_callback(lr_start, lr_max, lr_min, lr_ramp_ep, lr_sus_ep, lr_decay, batch_size=CFG.BATCH_SIZE)
    else:
        return tf.keras.optimizers.Adam(0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
    return optimizer