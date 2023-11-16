from tfswin import SwinTransformerV2Large256, preprocess_input
from tensorflow.keras import layers, models, Model, optimizers, losses, metrics
import wandb


def get_model(res: int = 256, num_classes: int = 467) -> Model:
    inputs = layers.Input(shape=(*res, 3), dtype='int8')
    outputs = SwinTransformerV2Large256(include_top=False, pooling='avg', input_shape=[*res, 3])(inputs)
    outputs = layers.Dense(num_classes, activation='softmax')(outputs)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_model(CFG, num_classes: int = 467):
    lr_rate = optimizers.schedules.CosineDecay(CFG.LR_START, warmup_steps=CFG.WARMUP_STEPS, decay_steps=CFG.DECAY_STEPS, alpha=CFG.ALPHA, warmup_target=CFG.WARMUP_TARGET)
    optimizer = optimizers.AdamW(lr_rate)
    loss = losses.SparseCategoricalCrossentropy()
    model = get_model(res=CFG.IMAGE_SIZE, num_classes=num_classes)
    model.compile(loss=loss, optimizer=optimizer, metrics=[
        "accuracy",
        metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy")
        ])
    return model

def get_callbacks(CFG):
    CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        wandb.keras.WandbMetricsLogger(log_freq="batch"),
        wandb.keras.WandbModelCheckpoint(
            str(CFG.CKPT_DIR),  # .h5 for weights, dir for whole model
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            # options=options,
            initial_value_threshold=2.0,

        ),    
    ]
    return callbacks
