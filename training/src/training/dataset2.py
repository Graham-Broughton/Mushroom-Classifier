from tfswin import preprocess_input
import re
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import math
from sklearn.model_selection import train_test_split


AUTO = tf.data.experimental.AUTOTUNE


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def decode_image(image_data, CFG):
    image = tf.image.decode_jpeg(image_data, channels=3)  # image format uint8 [0,255]
    image = tf.cast(image, tf.uint8)
    image = tf.image.resize_with_crop_or_pad(image, CFG.RAW_SIZE, CFG.RAW_SIZE)
    image = tf.image.random_crop(image, size=[*CFG.IMAGE_SIZE, 3])  #, method="lanczos5")
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
    dataset = dataset.map(lambda x: read_labeled_tfrecord(x, CFG), num_parallel_calls=AUTO)
    return dataset

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transform matrix which transforms indices

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.0
    shear = math.pi * shear / 180.0

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_mat([c1, s1, zero, -s1, c1, zero, zero, zero, one])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one, s2, zero, zero, c2, zero, zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one])
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one, zero, height_shift, zero, one, width_shift, zero, zero, one])

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))

def data_augment(img, label, CFG):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    # img = tf.image.stateless_random_crop(img, [CFG.MODEL_SIZE, CFG.MODEL_SIZE, 3])
    img = transform(img, CFG)
    img = tf.image.random_flip_left_right(img)
    # img = image.random_hue(img, 0.01)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_brightness(img, 0.1)
    return img, label

def transform(image, CFG):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = CFG.IMAGE_SIZE[0]
    XDIM = DIM % 2  # fix for size 331

    rot = 15. * tf.random.normal([1], dtype='float32')
    shr = 5. * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    h_shift = 16. * tf.random.normal([1], dtype='float32')
    w_shift = 16. * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])

def get_batched_dataset(filenames, CFG, train=False):
    dataset = load_dataset(filenames, CFG)
    dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTO)
    dataset = dataset.cache() # This dataset fits in RAM
    if train:
        if CFG.AUGMENT:
            dataset = dataset.map(lambda x, y: data_augment(x, y, CFG), num_parallel_calls=AUTO)
        # dataset = dataset.shuffle(BATCH_SIZE * 10)
        dataset = dataset.repeat()
    dataset = dataset.batch(CFG.BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    # should shuffle too but this dataset was well shuffled on disk already
    return dataset

def get_training_dataset(CFG):
    GCS_PATH_SELECT = {
        192: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-192x192",
        224: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-224x224v2",
        256: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-256x256",
        384: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-384x384",
        512: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-512x512",
        None: f"gs://{CFG.GCS_REPO}/tfrecords-jpeg-raw",
    }
    GCS_PATH = GCS_PATH_SELECT[CFG.RAW_SIZE]

    filenames = tf.io.gfile.glob(f"{GCS_PATH}/train*.tfrec")
    filenames, test_filenames = train_test_split(filenames, test_size=1)
    training_filenames, validation_filenames = train_test_split(filenames, test_size=0.2, shuffle=True)

    num_train = count_data_items(training_filenames)
    num_val = count_data_items(validation_filenames)

    return training_filenames, validation_filenames, test_filenames, num_train, num_val