import cv2
import pandas as pd
import tensorflow as tf

from config import CFG

AUTO = tf.data.experimental.AUTOTUNE


def load_dataframe(root_path):
    print(root_path)
    df = pd.read_csv(root_path / "train.csv")
    val = df[df["set"] == "val"].sample(frac=1, random_state=42).reset_index(drop=True)
    train = (
        df[df["set"] == "train"].sample(frac=1, random_state=42).reset_index(drop=True)
    )
    del df
    return train, val


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(
    feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7
):
    feature = {
        "image": bytes_feature(feature0),
        "dataset": int64_feature(feature1),
        "set": bytes_feature(feature2),
        "longitude": float_feature(feature3),
        "latitude": float_feature(feature4),
        "norm_date": float_feature(feature5),
        "class_priors": float_feature(feature6),
        "class_id": int64_feature(feature7),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_data(df, num_records=107):
    IMGS = df["file_path"].to_list()
    SIZE = len(IMGS) // num_records
    CT = len(IMGS) // SIZE + int(len(IMGS) % SIZE != 0)
    return IMGS, SIZE, CT


def write_records(
    df, set, tfrec_path, img_path, num_records, reshape_sizes
):
    print(tfrec_path, img_path, num_records, reshape_sizes)
    IMGS, SIZE, CT = get_data(df, num_records)
    for j in range(CT):
        print()
        print(f"Writing TFRecord {j} of {CT}...")
        CT2 = min(SIZE, len(IMGS) - j * SIZE)
        if not (tfrec_path / "tfrecords-jpeg-{reshape_sizes[0]}x{reshape_sizes[1]}").is_dir():
            (tfrec_path / "tfrecords-jpeg-{reshape_sizes[0]}x{reshape_sizes[1]}").mkdir(
                parents=True, exist_ok=True
            )
        with tf.io.TFRecordWriter(
            str(tfrec_path / "tfrecords-jpeg-{reshape_sizes[0]}x{reshape_sizes[1]}" / f"{set}{j:02d}-{CT2}.tfrec")
        ) as writer:
            for k in range(CT2):
                img = cv2.imread(str(img_path) + f"/{IMGS[SIZE * j + k]}")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(
                    img, reshape_sizes, cv2.INTER_CUBIC
                )  # Fix incorrect colors
                img = cv2.imencode(".jpg", img)[1].tobytes()
                row = df.loc[df.file_path == IMGS[SIZE * j + k]]
                example = serialize_example(
                    img,
                    row.dataset.values[0],
                    str.encode(row.set.values[0]),
                    row.longitude.values[0],
                    row.latitude.values[0],
                    row.norm_date.values[0],
                    row.class_priors.values[0],
                    row.class_id.values[0],
                )
                writer.write(example)
                if k % 100 == 0:
                    print(k, ", ", end="")


if __name__ == "__main__":
    import argparse
    CFG = CFG()

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d", "--img-directory",
        default=CFG.DATA,
        const=CFG.DATA,
        nargs="?",
        required=False,
        help="Image base location",
    )
    argparser.add_argument(
        "-p", "--tfrecords-directory",
        default=CFG.DATA,
        const=CFG.DATA,
        nargs="?",
        required=False,
        help="tfrecords wanted location",
    )
    argparser.add_argument(
        "-t", "--num-train-records",
        type=int,
        nargs="?",
        default=CFG.NUM_TRAINING_IMAGES,
        help="number of train records",
    )
    argparser.add_argument(
        "-v", "--num-validation-records",
        type=int,
        nargs="?",
        default=CFG.NUM_VALIDATION_IMAGES,
        help="number of validation records",
    )
    argparser.add_argument(
        "-s", "--img-size", 
        type=int, 
        nargs=2, 
        default=CFG.IMAGE_SIZE, 
        help="image size"
    )
    args = argparser.parse_args()

    train, val = load_dataframe(root_path=args.img_directory)
    for df, set in zip([train, val], ["train", "val"]):
        write_records(
            df,
            set,
            args.tfrecords_directory,
            args.img_directory,
            args.num_train_records if set == "train" else args.num_validation_records,
            reshape_sizes=args.img_size,
        )
