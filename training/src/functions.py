import matplotlib.pyplot as plt
import re
import math
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tfswin import SwinTransformerV2Large256, preprocess_input

class_dict = pickle.load(open("../class_dict.pkl", "rb"))
AUTO = tf.data.experimental.AUTOTUNE


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


def decode_image(image_data, CFG):
    image = tf.image.decode_jpeg(image_data, channels=3)  # image format uint8 [0,255]
    image = tf.cast(image, tf.uint8)
    image = tf.image.resize_with_crop_or_pad(image, *CFG.RAW_SIZE)
    image = tf.image.random_crop(
        image, size=[*CFG.CROP_SIZE, 3]
    )  # , method="lanczos5")
    return image


def read_labeled_tfrecord(example, CFG):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/id": tf.io.FixedLenFeature([], tf.string),
        "image/meta/dataset": tf.io.FixedLenFeature([], tf.int64),
        "image/meta/longitude": tf.io.FixedLenFeature([], tf.float32),
        "image/meta/latitude": tf.io.FixedLenFeature([], tf.float32),
        "image/meta/date": tf.io.FixedLenFeature([], tf.string),
        "image/meta/class_priors": tf.io.FixedLenFeature([], tf.float32),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        "image/class/text": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = decode_image(example["image/encoded"], CFG)
    label = tf.cast(example["image/class/label"], tf.int32)
    return image, label


def load_dataset(filenames, CFG):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(
        lambda x: read_labeled_tfrecord(x, CFG), num_parallel_calls=AUTO
    )
    return dataset


def get_model(res: int = [256, 256], num_classes: int = 467) -> tf.keras.Model:
    inputs = layers.Input(shape=(*res, 3), dtype="int8")
    outputs = SwinTransformerV2Large256(
        include_top=False, pooling="avg", input_shape=[*res, 3]
    )(inputs)
    outputs = layers.Dense(num_classes, activation="softmax")(outputs)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """Returns 3x3 transform matrix which transforms indices"""

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.0
    shear = math.pi * shear / 180.0

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    rotation_matrix = get_3x3_mat([c1, s1, zero, -s1, c1, zero, zero, zero, one])

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = get_3x3_mat([one, s2, zero, zero, c2, zero, zero, zero, one])

    zoom_matrix = get_3x3_mat(
        [one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one]
    )

    shift_matrix = get_3x3_mat(
        [one, zero, height_shift, zero, one, width_shift, zero, zero, one]
    )
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


def data_augment(img, label, CFG):
    # img = transform(img, CFG)
    img = tf.image.random_flip_left_right(img)
    # img = image.random_hue(img, 0.01)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_brightness(img, 0.1)
    return img, label


def transform(image, CFG):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = CFG.CROP_SIZE[0]
    XDIM = DIM % 2  # fix for size 331

    rot = 15.0 * tf.random.normal([1], dtype="float32")
    shr = 5.0 * tf.random.normal([1], dtype="float32")
    h_zoom = 1.0 + tf.random.normal([1], dtype="float32") / 10.0
    w_zoom = 1.0 + tf.random.normal([1], dtype="float32") / 10.0
    h_shift = 16.0 * tf.random.normal([1], dtype="float32")
    w_shift = 16.0 * tf.random.normal([1], dtype="float32")

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])


def get_batched_dataset(filenames, CFG, train=False):
    dataset = load_dataset(filenames, CFG)
    dataset = dataset.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTO
    )
    dataset = dataset.cache()  # This dataset fits in RAM
    if train:
        if CFG.AUGMENT:
            dataset = dataset.map(
                lambda x, y: data_augment(x, y, CFG), num_parallel_calls=AUTO
            )
        # dataset = dataset.shuffle(BATCH_SIZE * 10)
        dataset = dataset.repeat()
    dataset = dataset.batch(CFG.BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(
        AUTO
    )  # prefetch next batch while training (autotune prefetch buffer size)
    # should shuffle too but this dataset was well shuffled on disk already
    return dataset


def dataset_to_numpy_util(dataset, N):
    dataset = dataset.batch(N)

    # In eager mode, iterate in the Datset directly.
    for images, labels in dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        break

    return numpy_images, numpy_labels


def display_one_flower(image, title, subplot, red=False):
    plt.subplot(subplot)
    plt.axis("off")
    plt.imshow(image)
    plt.title(title, fontsize=16, color="red" if red else "black")
    return subplot + 1


def display_9_images_from_dataset(dataset):
    subplot = 331
    plt.figure(figsize=(13, 13))
    images, labels = dataset_to_numpy_util(dataset, 9)
    for i, image in enumerate(images):
        title = class_dict[np.argmax(labels[i], axis=-1)]
        subplot = display_one_flower(image, title, subplot)
        if i >= 8:
            break

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
