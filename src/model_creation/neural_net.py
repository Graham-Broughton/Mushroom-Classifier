import tensorflow as tf
from tensorflow import keras

def make_neural_network(base_arch_name, weights, image_size, dropout_pct, n_classes, input_dtype, train_full_network):
    image_size_with_channels = IMAGE_SIZE + [3]
    base_arch = base_arch_name


    input_layer = keras.layers.Input(shape=image_size_with_channels, dtype=input_dtype)
    base_model = base_arch(input_tensor=input_layer, weights=weights, include_top=False)
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    dropout = keras.layers.Dropout(dropout_pct)(avg)
    x = keras.layers.Dense(NUM_CLASSES, name="dense_logits")(dropout)
    output = keras.layers.Activation("sigmoid", dtype="float32", name="predictions")(x)
    model = keras.Model(inputs=base_model.input, outputs=output)


    if train_full_network:
        for layer in base_model.layers:
            layer.trainable = True

    return model
