import warnings
from os import environ
from pathlib import Path

import cv2
import pandas as pd
from loguru import logger
from tqdm import trange
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from training.train_config import CFG

CFG = CFG()
warnings.filterwarnings("ignore")

AUTO = tf.data.experimental.AUTOTUNE


def load_dataframe(path: Path):
    """This function loads a dataframe from the given root path.

    Parameters:
        path (pathlib.Path): The root path where the dataframe is located.

    Returns:
        DataFrame: The loaded dataframe.
    """
    df = pd.read_csv(path)
    df = (
        df.sample(frac=1.0, random_state=CFG.SEED)
        .sample(frac=1.0, random_state=CFG.SEED)
        .reset_index(drop=True)
    )

    logger.info("Loaded and shuffled dataframe")
    logger.debug(f"Dataframe shape: {df.shape}")
    return df


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
    feature0: str,
    feature1: str,
    feature2: int,
    feature3: int,
    feature4: int,
):
    """This function serializes an example with given features.

    Returns:
        str: Serialized example.
    """
    feature = {
        "image/encoded": bytes_feature(feature0),
        "image/id": bytes_feature(feature1),
        "image/meta/width": int64_feature(feature2),
        "image/meta/height": int64_feature(feature3),
        "image/class/label": int64_feature(feature4),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_data(df: pd.DataFrame, num_records: int = 107):
    """This function gets data from a dataframe to use in calculating tfrecord attributes.

    Parameters:
        df (DataFrame): The dataframe to get data from.
        num_records (int, optional): The number of records to get. Defaults to 107.

    Returns:
        tuple: A tuple containing the images, size, and count.
    """
    IMGS = df["file_path"].to_list()
    SIZE = len(IMGS) // num_records
    CT = len(IMGS) // SIZE + int(len(IMGS) % SIZE != 0)
    logger.debug(f"Number of records: {num_records}, size: {SIZE}, count: {CT}")
    return IMGS, SIZE, CT


def write_sp_tfrecords(
    dataframe_path: Path,
    tfrec_path: Path,
    num_records: int,
):
    """This function writes single process TFRecords.

    Parameters:
        dataframe_path (pathlib.Path): The path to the dataframe.
        tfrec_path (pathlib.Path): The path to write the TFRecords.
        num_records (int): The number of records.
    """
    # logger.info("Writing TFRecords single process")
    ## TODO: Add tqdm functionality in loguru
    # tqdm_logger = logger.add(lambda msg: tqdm.write(msg, end=""))

    # load dataframes and iterate over them
    df = load_dataframe(path=dataframe_path)
    IMGS, SIZE, CT = get_data(df, num_records)

    # iterate over the number of tfrecords
    for j in trange(CT):
        # logger.info(f"Writing {j:02d} of {CT} tfrecords")
        CT2 = min(SIZE, len(IMGS) - j * SIZE)  # get the number of images in a tfrecord

        # create the path to write the tfrecord to
        path = tfrec_path / "tfrecords-jpeg-raw"
        path.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(str(path / f"train{j:02d}-{CT2}.tfrec")) as writer:
            # Iterate through the rows of the dataframe
            for k in trange(CT2, leave=False):
                row = df[df["file_path"] == IMGS[SIZE * j + k]].iloc[0]
                # load image from disk, resize to reshape_sizes and encode as jpeg
                img = cv2.imread(row.file_path)
                img = cv2.imencode(".jpg", img)[1].tobytes()

                # Serialize data
                example = serialize_example(
                    img,
                    str(row.file_name).split(".")[0].encode("utf8"),
                    row.width,
                    row.height,
                    row.class_id,
                )
                writer.write(example)


def get_mp_params(tfrec_dir: Path, num_records: int, df: pd.DataFrame):
    """Returns a list of dictionaries containing parameters for creating TensorFlow Records.

    Args:
        tfrec_dir (pathlib.Path): The path to the directory where the TensorFlow Records will be saved.
        num_records (int): The number of records to create.
        df (DataFrame): A pandas DataFrame containing the image filenames and labels.

    Returns:
        list: A list of dictionaries containing the parameters for creating TensorFlow Records.
    """
    sliced_dfs = []  # multiprocessing requires the arguments to be in a list
    tfrec_names = []
    for i in range(num_records):
        n = len(df) // num_records  # number of images in each tfrecord
        cnt = min(n, df.shape[0] - i * n)  # find the number of images in the tfrecord

        # create the path to write the tfrecord to
        if not tfrec_dir.is_dir():
            tfrec_dir.mkdir(parents=True, exist_ok=True)

        sliced_dfs.append(df.iloc[(i * n) : (i * n + cnt)])
        tfrec_names.append(str(tfrec_dir / f"train{i:02d}-{cnt}.tfrec"))
    return sliced_dfs, tfrec_names


def write_single_mp_tfrecord(params: tuple):
    """Writes a single TensorFlow record file from a given dset of parameters.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing the dataset.
        path (str): The path to write the TensorFlow record file to.
    """
    # loading the parameters from param dictionary
    df, tfrec_name = params
    df = df

    with tf.io.TFRecordWriter(tfrec_name) as writer:
        # TODO: Add tqdm functionality in loguru
        for _, row in df.iterrows():
            image_path = row["file_path"]  # for each image in the tfrecord..
            # load image from disk, change RGB to cv2 default BGR format, resize to reshape_sizes and encode as jpeg
            img = cv2.imread(image_path)
            img = cv2.imencode(".jpg", img)[1].tobytes()

            # read specific row from dataframe to get data and serialize it
            example = serialize_example(
                img,
                str(row.file_name).split(".")[0].encode("utf-8"),
                row.width,
                row.height,
                row.class_id,
            )
            writer.write(example)


def write_mp_tfrecords(dataframe_path, tfrecords_directory, num_records, num_processes=cpu_count()):
    """Writes multiple TFRecord files for the mushroom classifier dataset.

    Args:
        dataframe_path (pathlib.Path): Path to the dataframe.
        tfrecords_directory (pathlib.Path): Path to the directory where the TFRecord files will be saved.
        num_train_records (int): Number of training records to write.
    """
    df = load_dataframe(dataframe_path)

    sliced_dfs, tfrec_names = get_mp_params(
        tfrecords_directory,
        num_records,
        df,
    )
    logger.info("Starting multiprocessing tfrecords")
    # use multiprocessing to write the tfrecords
    process_map(
        write_single_mp_tfrecord,
        list(zip(sliced_dfs, tfrec_names)),
        max_workers=num_processes,
        leave=True,
    )
    logger.info("Finished multiprocessing tfrecords")


if __name__ == "__main__":
    import argparse  # only needed when called as script

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # argparse to allow for command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d",
        "--df-path",
        required=True,
        help="Path to dataframe",
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
        "-r",
        "--num-records",
        type=int,
        nargs="?",
        default=CFG.NUM_TRAINING_RECORDS,
        help="number of tfrecords records to create",
    )
    argparser.add_argument(
        "-m",
        "--multiprocessing",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="whether to use multiprocessing",
    )
    argparser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        nargs="?",
        default=cpu_count(),
        help="number of processes to use",
    )
    args = argparser.parse_args()

    # If statement to use multiprocessing if flag is set
    if args.multiprocessing:
        logger.info("Writing TFRecords multiprocessing")
        write_mp_tfrecords(
            dataframe_path=Path(args.df_path),
            tfrecords_directory=Path(args.tfrecords_directory),
            num_records=args.num_records,
            num_processes=args.num_processes,
        )
    else:
        logger.info("Writing TFRecords single process")
        write_sp_tfrecords(
            Path(args.df_path),
            args.tfrecords_directory,
            args.num_train_records,
        )
