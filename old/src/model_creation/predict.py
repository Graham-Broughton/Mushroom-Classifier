import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

def make_prediction(img, img_size):
    """
    takes a file path, and image size as input and returns an array of probabilities
    as well as the highest proba
    """
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.expand_dims(img, axis=0)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size)
    
    return model.predict(img), np.argmax(model.predict(img))
