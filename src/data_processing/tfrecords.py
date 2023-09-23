import tensorflow as tf
import cv2


AUTO = tf.data.experimental.AUTOTUNE


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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
        'dataset': int64_feature(feature1),
        'set': bytes_feature(feature2),
        'longitude': float_feature(feature3),
        'latitude': float_feature(feature4),
        'norm_date': float_feature(feature5),
        'target': int64_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_records(df, CT, SIZE, IMGS, set, path):
    for j in range(CT):
        print()
        print(f'Writing TFRecord {j} of {CT}...')
        CT2 = min(SIZE, len(IMGS) - j * SIZE)
        with tf.io.TFRecordWriter(f'{str(path)}/tfrec/' + f'{set}{j:02d}-{CT2}.tfrec') as writer:
            for k in range(CT2):
                img = cv2.imread(f'../{IMGS[SIZE * j + k]}')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Fix incorrect colors
                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tobytes()
                row = df.loc[df.file_path == IMGS[SIZE * j + k]]
                example = serialize_example(
                    img,
                    row.dataset.values[0],
                    str.encode(row.set.values[0]),
                    row.longitude.values[0],
                    row.latitude.values[0],
                    row.norm_date.values[0],
                    row.class_id.values[0],
                )
                writer.write(example)
                if k % 100 == 0:
                    print(k, ', ', end='')
