import tensorflow as tf
from tensorflow import keras
from tf.keras import layers, applications

def _model(shape, dropout_pct, classes, train_layers=False):
    input = layers.Input(shape=shape+[3], dtype=tf.float32)
    base = applications.efficientnet_v2.EfficientNetV2B0(include_top=False, input_tensor=input)

    dropout = layers.Dropout(dropout_pct)(base.output)
    pool = layers.GlobalAveragePooling2D()(dropout)
    #x = layers.Dense(2048, activation='swish')(avg)
    x = layers.Dense(classes, activation='softmax')(pool)

    model = keras.Model(inputs=base.input, outputs=x)

    if train_layers == True:
        for layer in base.layers:
            layer.trainable=True
    return model

def create_model(lr, shape, dropout_pct, classes):
    #optimizer = tfa.optimizers.AdamW(0.001, 0.001)
    optimizer = tf.keras.optimizers.Nadam(lr)

    model = _model(shape, dropout_pct, classes)

    model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3 accuracy')])
    
    return model