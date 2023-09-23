import tensorflow as tf
import tensorflow.keras.backend as K
import math


AUTO = tf.data.experimental.AUTOTUNE


def read_labeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'dataset': tf.io.FixedLenFeature([], tf.int64),
        'set': tf.io.FixedLenFeature([], tf.string),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'norm_date': tf.io.FixedLenFeature([], tf.float32),
        'target': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['target']


def read_unlabeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'norm_date': tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image']


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


def transform(image, CFG, DIM=256):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    XDIM = DIM % 2  # fix for size 331

    rot = CFG.ROT_ * tf.random.normal([1], dtype='float32')
    shr = CFG.SHR_ * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / CFG.HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / CFG.WZOOM_
    h_shift = CFG.HSHIFT_ * tf.random.normal([1], dtype='float32')
    w_shift = CFG.WSHIFT_ * tf.random.normal([1], dtype='float32')

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


def prepare_image(img, CFG, augment=True, dim=256):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = transform(img, CFG, DIM=dim)
        img = tf.image.random_flip_left_right(img)
        # img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    img = tf.reshape(img, [dim, dim, 3])

    return img


def get_dataset(
    files, CFG, augment=False, shuffle=False, repeat=False, labeled=True, batch_size=16, dim=256):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example), num_parallel_calls=AUTO)

    ds = ds.map(
        lambda img, imgname_or_label: (prepare_image(
            img, CFG, augment=augment, dim=dim), imgname_or_label), num_parallel_calls=AUTO
    )

    ds = ds.batch(batch_size * CFG.REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds
