import tensorflow as tf
import tensorflow.keras.backend as K
import math

AUTO = tf.data.experimental.AUTOTUNE


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(feature0, feature1, feature2, feature3, feature4, feature5, feature6):
    feature = {
        'image': bytes_feature(feature0),
        'longitude': float_feature(feature1),
        'latitude': float_feature(feature2),
        'data_normed': float_feature(feature3),
        'dataset': float_feature(feature4),
        'set': bytes_feature(feature5),
        'target': int64_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# active, VERBOSE=2 for commit
# from dotenv import load_dotenv, set_key
# from sklearn.model_selection import KFold
# import numpy as np
# import tensorflow.keras.backend as
