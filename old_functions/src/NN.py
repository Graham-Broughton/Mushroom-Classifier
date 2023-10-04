import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, applications, layers, metrics, optimizers
import tensorflow_hub as hub
import os


os.environ["TFHUB_CACHE_DIR"] = "gs://mush-img-repo/models/cache"


def _model(shape, dropout_pct, classes, train_layers=False):
    inputs = layers.Input(shape=shape + [3], dtype=tf.float32)
    base = applications.efficientnet_v2.EfficientNetV2B0(include_top=False, input_tensor=inputs)

    dropout = layers.Dropout(dropout_pct)(base.output)
    pool = layers.GlobalAveragePooling2D()(dropout)
    # x = layers.Dense(2048, activation='swish')(avg)
    x = layers.Dense(classes, activation='softmax')(pool)

    model = Model(inputs=inputs, outputs=x)

    if train_layers:
        for layer in base.layers:
            layer.trainable = True
    return model


def _build_model(model, num_classes, dim=128):
    inp = tf.keras.layers.Input(shape=(dim,dim,3))
    base = hub.load(model, trainable=True)#(input_shape=(dim,dim,3),weights='imagenet',include_top=False)
    x = base(inp, training=True)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, outputs)
    return model


def create_model(model_path, lr, shape, dropout_pct, classes):
    optimizer = tfa.optimizers.AdamW(0.001, 0.001)
    # optimizer = optimizers.Nadam(lr)

    model = _build_model(model_path, classes, dim=shape[0])

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', metrics.SparseTopKCategoricalAccuracy(k=3, name='top3 accuracy')],
    )
    return model
