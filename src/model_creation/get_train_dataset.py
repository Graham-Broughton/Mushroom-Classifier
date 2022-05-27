from functools import partial
import pandas as pd
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

# augments
def _decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # no resizing, we may augment which will crop.
    # we resize _after_ the augments pass
    return img


def process_row(file_path, label, num_classes):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = _decode_img(img)
    # 1 hot encode the label for dense
    label = tf.one_hot(label, num_classes)
    return img, label


def _load_dataframe(dataset_json_path):
    df = pd.read_csv(dataset_json_path)

    # sort the dataset
    df = df.sample(frac=1, random_state=42)
    return df


def _prepare_dataset(
    ds,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    repeat_forever=True,
    shuffle_buffer_size=4000,
    augment=False,
):
    # shuffle
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # resize to image size expected by network
    ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y))

    # Repeat forever
    if repeat_forever:
        ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def make_dataset(
    path,
    label_column_name,
    image_column_name
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    repeat_forever=True,
    augment=False,
):
    df = _load_dataframe(path)
    num_examples = len(df)
    num_classes = len(df[label_column_name].unique())

    ds = tf.data.Dataset.from_tensor_slices((df[image_column_name], df[label_column_name]))

    process_partial = partial(process_row, num_classes=num_classes)
    ds = ds.map(process_partial, num_parallel_calls=AUTOTUNE)

    ds = _prepare_dataset(
        ds,
        image_size=image_size,
        batch_size=batch_size,
        repeat_forever=repeat_forever,
        augment=augment,
    )

    return (ds, num_examples)
