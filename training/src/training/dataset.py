import tensorflow as tf
import tensorflow_addons as tfa

AUTO = tf.data.experimental.AUTOTUNE


def decode_image(image_data, smallest_side, CFG):
    image = tf.image.decode_jpeg(image_data, channels=3)  # image format uint8 [0,255]
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, smallest_side, smallest_side)
    image = tf.image.resize(image, size=CFG.RAW_SIZE, method="lanczos5")
    image = tf.image.random_crop(image, size=[*CFG.CROP_SIZE, 3])  # , method="lanczos5"
    return image


def read_labeled_tfrecord(example, CFG):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/id": tf.io.FixedLenFeature([], tf.string),
        "image/meta/longitude": tf.io.FixedLenFeature([], tf.float32),
        "image/meta/latitude": tf.io.FixedLenFeature([], tf.float32),
        "image/meta/month": tf.io.FixedLenFeature([], tf.int64),
        "image/meta/day": tf.io.FixedLenFeature([], tf.int64),
        "image/meta/class_priors": tf.io.FixedLenFeature([], tf.float32),
        "image/meta/width": tf.io.FixedLenFeature([], tf.int64),
        "image/meta/height": tf.io.FixedLenFeature([], tf.int64),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)

    width = tf.cast(example["image/meta/width"], tf.int32)
    height = tf.cast(example["image/meta/height"], tf.int32)
    smallest_side = tf.minimum(width, height)

    image = decode_image(example["image/encoded"], smallest_side, CFG)
    label = tf.cast(example["image/class/label"], tf.int32)

    month = tf.cast(example["image/meta/month"], tf.float32)
    day = tf.cast(example["image/meta/day"], tf.float32)

    lon = tf.cast(example["image/meta/longitude"], tf.float32)
    lat = tf.cast(example["image/meta/latitude"], tf.float32)
    latlong = tf.stack([lat, lon], axis=-1)  # Convert to a tensor

    id = tf.cast(example["image/id"], tf.string)

    meta = dict(latlong=latlong, month=month, day=day)  # , id=id
    return image, meta, label


def get_date_feats(meta):
    days = meta["day"]
    months = meta["month"]

    months_in_year = 12
    days_in_year = 365

    day_sin = tf.math.sin(days * (2 * np.pi / days_in_year))
    day_cos = tf.math.cos(days * (2 * np.pi / days_in_year))

    month_sin = tf.math.sin(months * (2 * np.pi / months_in_year))
    month_cos = tf.math.cos(months * (2 * np.pi / months_in_year))
    return tf.stack([day_sin, day_cos, month_sin, month_cos], axis=-1)


def map_coordinates_to_grid(meta, grid_size=(100, 100), normalize=True):
    latitudes, longitudes = tf.unstack(meta["latlong"], axis=-1)

    normalized_lat = (latitudes + 90) / 180
    normalized_lon = (longitudes + 180) / 360

    grid_x = tf.cast(normalized_lon * grid_size[1], tf.float32)
    grid_y = tf.cast(normalized_lat * grid_size[0], tf.float32)

    grid_x = tf.clip_by_value(grid_x, 0, grid_size[1] - 1)
    grid_y = tf.clip_by_value(grid_y, 0, grid_size[0] - 1)

    if normalize:
        # Normalize to 0-1 range
        grid_x = grid_x / (grid_size[1] - 1)
        grid_y = grid_y / (grid_size[0] - 1)

    return tf.stack([grid_x, grid_y], axis=-1)


@tf.function
def process_meta(meta):
    date_feats = get_date_feats(meta)
    gps_feat = map_coordinates_to_grid(meta)
    meta = tf.concat([date_feats, gps_feat], axis=-1)
    return meta


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


def basic_augment(
    image,
    prob_flip=0.5,
    prob_color_jitter=0.5,
    prob_rotate=0.5,
):
    """Apply various augmentations to an image.

    Args:
        image: The input image.
        prob_flip: Probability of applying horizontal flip.
        prob_color_jitter: Probability of applying color jitter.
        prob_rotate: Probability of applying rotation.

    Returns:
        Augmented image.
    """
    # Horizontal Flip
    if tf.random.uniform([]) < prob_flip:
        image = tf.image.random_flip_left_right(image)

    # Color Jitter
    if tf.random.uniform([]) < prob_color_jitter:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.1)

    # Rotation
    if tf.random.uniform([]) < prob_rotate:
        angles = tf.random.uniform([], -15, 15)
        image = tfa.image.rotate(image, angles)
    return image


def random_masking(image, randmask_prob=0.5):
    if tf.random.uniform([]) < randmask_prob:
        original_shape = tf.shape(image)

        count = tf.random.uniform([], 1, 11)

        erase_size = tf.random.uniform([], 0.15, 0.45)
        erase_size = tf.math.divide(erase_size, tf.math.sqrt(count))
        erase_value = tf.random.uniform([], 0.0, 255.0)

        mask1 = int(erase_size * float(original_shape[0])) - (
            int(erase_size * float(original_shape[0])) % 2
        )
        mask2 = int(erase_size * float(original_shape[1])) - (
            int(erase_size * float(original_shape[1])) % 2
        )
        for k in tf.range(count):
            image = tfa.image.random_cutout(
                tf.expand_dims(image, axis=0),
                mask_size=(mask1, mask2),
                constant_values=erase_value,
            )
            image = tf.squeeze(image, 0)
    return image


def augment(images, meta, labels):
    images = tf.map_fn(basic_augment, images)
    images = tf.map_fn(random_masking, images)
    return images, meta, labels


def get_batched_dataset(filenames, CFG, train=False):
    dataset = load_dataset(filenames, CFG)
    dataset = dataset.cache()  # This dataset fits in RAM
    dataset = dataset.batch(CFG.BATCH_SIZE, drop_remainder=True)
    if train:
        dataset = dataset.map(augment, num_parallel_calls=AUTO)
        # dataset = dataset.shuffle(BATCH_SIZE * 10)
        # dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x, y, z: ((x, process_meta(y)), z), num_parallel_calls=AUTO
    )
    dataset = dataset.prefetch(
        AUTO
    )  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
