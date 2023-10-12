from os import environ
from loguru import logger
import cv2
import pandas as pd
import warnings
from tqdm import trange

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from config import CFG

warnings.filterwarnings("ignore")

AUTO = tf.data.experimental.AUTOTUNE


def load_dataframe(root_path):
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
    feature0, feature1, feature2, feature3, feature4, feature5, feature6
):
    feature = {
        "image": bytes_feature(feature0),
        "dataset": int64_feature(feature1),
        "longitude": float_feature(feature2),
        "latitude": float_feature(feature3),
        "norm_date": float_feature(feature4),
        "class_priors": float_feature(feature5),
        "class_id": int64_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_data(df, num_records=107):
    IMGS = df["file_path"].to_list()
    SIZE = len(IMGS) // num_records
    CT = len(IMGS) // SIZE + int(len(IMGS) % SIZE != 0)
    return IMGS, SIZE, CT


def write_records(df, set, tfrec_path, img_path, num_records, reshape_sizes):
    IMGS, SIZE, CT = get_data(df, num_records)
    for j in trange(CT):
        CT2 = min(SIZE, len(IMGS) - j * SIZE)
        path = tfrec_path / f"tfrecords-jpeg-{reshape_sizes[0]}x{reshape_sizes[1]}"
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        with tf.io.TFRecordWriter(str(path / f"{set}{j:02d}-{CT2}.tfrec")) as writer:
            for k in trange(CT2, leave=False):
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


def get_mp_params(tfrec_path, img_path, num_records, reshape_sizes, set, df):

    IMGS, SIZE, CT = get_data(df, num_records)
    param_list = []
    for j in range(CT):
        CT2 = min(SIZE, len(IMGS) - j * SIZE)
        path = tfrec_path / f"tfrecords-jpeg-{reshape_sizes[0]}x{reshape_sizes[1]}"
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        image_paths = []
        image_filenames = []
        for k in range(CT2):
            img_filename = IMGS[SIZE * j + k]
            image_filenames.append(img_filename)
            image_path = str(img_path / img_filename)
            image_paths.append(image_path)
        params = {
            "df": df,
            "path": str(path / f"{set}{j:02d}-{CT2}.tfrec"),
            "reshape_sizes": reshape_sizes,
            "image_paths": image_paths,
            "image_filenames": image_filenames,
        }
        param_list.append(params)
    return param_list


def write_mp(**kwargs):
    from tqdm.contrib.concurrent import process_map
    train, val = load_dataframe(root_path=kwargs['img_directory'])
    plist = []
    for df, set in zip([train, val], ["train", "val"]):
        params = get_mp_params(
            kwargs['tfrecords_directory'],
            kwargs['img_directory'],
            kwargs['num_train_records'] if set == "train" else kwargs['num_validation_records'],
            kwargs['img_size'],
            set,
            df,
        )
        plist.append(params)
    for params in plist:
        process_map(write_tfrecord_mp, params, max_workers=8)


def write_tfrecord_mp(params):
    df = params["df"]
    path = params["path"]
    reshape_sizes = params["reshape_sizes"]
    image_paths = params["image_paths"]
    image_filenames = params["image_filenames"]

    with tf.io.TFRecordWriter(path) as writer:
        for idx, (image_path, image_filename) in enumerate(zip(image_paths, image_filenames)):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(
                img, reshape_sizes, cv2.INTER_CUBIC
            )  # Fix incorrect colors
            img = cv2.imencode(".jpg", img)[1].tobytes()
            row = df.loc[df.file_path == image_filename]
            example = serialize_example(
                img,
                row.dataset.values[0],
                row.longitude.values[0],
                row.latitude.values[0],
                row.norm_date.values[0],
                row.class_priors.values[0],
                row.class_id.values[0],
            )
            writer.write(example)


if __name__ == "__main__":
    import argparse

    CFG = CFG()

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d",
        "--img-directory",
        default=CFG.DATA,
        const=CFG.DATA,
        nargs="?",
        required=False,
        help="Image base location",
    )
    argparser.add_argument(
        "-p",
        "--tfrecords-directory",
        default=CFG.DATA,
        const=CFG.DATA,
        nargs="?",
        required=False,
        help="tfrecords wanted location",
    )
    argparser.add_argument(
        "-t",
        "--num-train-records",
        type=int,
        nargs="?",
        default=CFG.NUM_TRAINING_IMAGES,
        help="number of train records",
    )
    argparser.add_argument(
        "-v",
        "--num-validation-records",
        type=int,
        nargs="?",
        default=CFG.NUM_VALIDATION_IMAGES,
        help="number of validation records",
    )
    argparser.add_argument(
        "-s", "--img-size", type=int, nargs=2, default=CFG.IMAGE_SIZE, help="image size"
    )
    argparser.add_argument(
        "-m",
        "--multiprocessing",
        action="store_false",
        help="whether to use multiprocessing",
    )
    args = argparser.parse_args()
    print(args.img_directory)
    if not args.multiprocessing:
        # d = vars(args)
        write_mp(
            tfrecords_directory = args.tfrecords_directory,
            img_directory = args.img_directory,
            num_train_records = args.num_train_records,
            num_validation_records = args.num_validation_records,
            img_size = args.img_size,
        )
    else:
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
