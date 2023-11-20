import warnings
from os import environ
from pathlib import Path

import cv2
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from tqdm import trange
from tqdm.contrib.concurrent import process_map

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from training.train_config import CFG

CFG = CFG()
warnings.filterwarnings("ignore")

AUTO = tf.data.experimental.AUTOTUNE


def load_dataframe(root_path: Path):
    """This function loads a dataframe from the given root path.

    Parameters:
        root_path (pathlib.Path): The root path where the dataframe is located.

    Returns:
        DataFrame: The loaded dataframe.
    """
    df = pd.read_csv(root_path.parent / "train.csv")
    df = df.sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)

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
    feature2: float,
    feature3: float,
    feature4: int,
    feature5: int,
    feature6: float,
    feature7: int,
    feature8: int,
    feature9: int,
):
    """This function serializes an example with given features.

    Returns:
        str: Serialized example.
    """
    feature = {
        "image/encoded": bytes_feature(feature0),
        "image/id": bytes_feature(feature1),
        "image/meta/longitude": float_feature(feature2),
        "image/meta/latitude": float_feature(feature3),
        "image/meta/month": int64_feature(feature4),
        "image/meta/day": int64_feature(feature5),
        "image/meta/class_priors": float_feature(feature6),
        "image/meta/width": int64_feature(feature7),
        "image/meta/height": int64_feature(feature8),
        "image/class/label": int64_feature(feature9),
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
    tfrec_path: Path,
    img_path: Path,
    num_train_records: int,
    # num_val_records: int,
    reshape_size: int,
):
    """This function writes single process TFRecords.

    Parameters:
        tfrec_path (pathlib.Path): The path to write the TFRecords.
        img_path (pathlib.Path): The path of the images.
        num_records (int): The number of records.
        reshape_sizes (tuple): The sizes to reshape the images.
    """
    logger.info("Writing TFRecords single process")
    ## TODO: Add tqdm functionality in loguru
    # tqdm_logger = logger.add(lambda msg: tqdm.write(msg, end=""))

    # load dataframes and iterate over them
    df = load_dataframe(root_path=img_path)
    num_records = num_train_records
    IMGS, SIZE, CT = get_data(df, num_records)

    # iterate over the number of tfrecords
    for j in trange(CT):
        logger.info(f"Writing {j:02d} of {CT} tfrecords")
        CT2 = min(
            SIZE, len(IMGS) - j * SIZE
        )  # get the number of images in a tfrecord

        # create the path to write the tfrecord to
        path = tfrec_path / f"tfrecords-jpeg-{reshape_size}x{reshape_size}"
        path.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(
            str(path / f"train{j:02d}-{CT2}.tfrec")
        ) as writer:
            for k in trange(CT2, leave=False):  # for each image in the tfrecord
                if k % 100 == 0:
                    logger.info(f"Writing {k:02d} of {CT2} train tfrecord images")

                # load image from disk, change RGB to cv2 default BGR format, resize to reshape_sizes and encode as jpeg
                img = cv2.imread(str(img_path / f"{IMGS[SIZE * j + k]}"))
                img = cv2.imencode(".jpg", img)[1].tobytes()

                # read specific row from dataframe to get data and serialize it
                row = df.loc[df.file_path == IMGS[SIZE * j + k]].iloc[0]
                example = serialize_example(
                    img,
                    row.file_name.split(".")[0],
                    row.longitude,
                    row.latitude,
                    row.date,
                    row.class_priors,
                    row.width,
                    row.height,
                    row.class_id,
                )
                writer.write(example)

def get_mp_params(
    tfrec_path: Path, num_records: int, reshape_size: int, df: pd.DataFrame
):
    """Returns a list of dictionaries containing parameters for creating TensorFlow Records.

    Args:
        tfrec_path (pathlib.Path): The path to the directory where the TensorFlow Records will be saved.
        img_path (pathlib.Path): The path to the directory containing the images.
        num_records (int): The number of records to create.
        reshape_size (tuple): A tuple containing the desired width and height of the images.
        dset (str): A string indicating the type of data (e.g. "train", "test", "validation").
        df (DataFrame): A pandas DataFrame containing the image filenames and labels.

    Returns:
        list: A list of dictionaries containing the parameters for creating TensorFlow Records.
    """
    skf = StratifiedKFold(n_splits=num_records, shuffle=True, random_state=CFG.SEED)
    param_list = []  # multiprocessing requires the arguments to be in a list
    for fold, (_, v_idx) in enumerate(
        skf.split(df, df["class_id"])
    ):  # for each tfrecord
        n = len(v_idx)
        cnt = min(
            n, df.shape[0] - fold * n
        )  # find the number of images in the tfrecord

        # create the path to write the tfrecord to
        path = tfrec_path / f"tfrecords-jpeg-{reshape_size}x{reshape_size}"
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        image_paths = []
        images = df.iloc[v_idx]
        image_paths = images["file_path"].to_list()

        # once the lists are saved, create a dictionary of parameters and append to the param list
        params = {
            "df": df.iloc[v_idx],
            "path": str(path / f"train{fold:02d}-{cnt}.tfrec"),
            "reshape_size": reshape_size,
            "image_paths": image_paths,
            "indices": v_idx,
        }
        param_list.append(params)
    return param_list


def write_single_mp_tfrecord(params):
    """Writes a single TensorFlow record file from a given dset of parameters.

    Args:
        params (dict): A dictionary containing the following keys:
            df (pandas.DataFrame): A pandas DataFrame containing the dataset.
            path (str): The path to write the TensorFlow record file to.
            reshape_sizes (tuple): A tuple containing the desired image reshape sizes.
            image_paths (list): A list of image paths.
            image_filenames (list): A list of image filenames.

    """
    # loading the parameters from param dictionary
    df = params["df"]
    path = params["path"]
    reshape_size = params["reshape_size"]

    with tf.io.TFRecordWriter(path) as writer:
        # TODO: Add tqdm functionality in loguru
        for _, row in df.iterrows():
            image_path = row["file_path"]  # for each image in the tfrecord..
            # load image from disk, change RGB to cv2 default BGR format, resize to reshape_sizes and encode as jpeg
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(
                img, (reshape_size, reshape_size), cv2.INTER_CUBIC
            )  # Fix incorrect colors
            img = cv2.imencode(".jpg", img)[1].tobytes()

            # read specific row from dataframe to get data and serialize it
            example = serialize_example(
                img,
                row.file_name.split(".")[0].encode("utf-8"),
                row.dataset,
                row.longitude,
                row.latitude,
                row.date.encode("utf-8"),
                row.class_priors,
                row.class_id,
                f"{row.genus}_{row.specific_epithet}".encode("utf-8"),
            )
            writer.write(example)


def write_mp_tfrecords(**kwargs):
    """Writes multiple TFRecord files for the mushroom classifier dataset.

    Args:
        **kwargs: Keyword arguments containing the following:
            img_directory (pathlib.Path): Path to the directory containing the images.
            tfrecords_directory (pathlib.Path): Path to the directory where the TFRecord files will be saved.
            num_train_records (int): Number of training records to write.
            num_validation_records (int): Number of validation records to write.
            img_size (tuple): Tuple containing the size of the images.
    """
    df = load_dataframe(root_path=kwargs["img_directory"])

    params = get_mp_params(
        kwargs["tfrecords_directory"],
        kwargs["num_train_records"],
        kwargs["img_size"],
        df,
    )
    logger.info("Starting multiprocessing tfrecords")
    # use multiprocessing to write the tfrecords
    process_map(write_single_mp_tfrecord, params, max_workers=8, leave=True)
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
        "--img-directory",
        default=CFG.RAW_DATA,
        const=CFG.RAW_DATA,
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
        default=CFG.NUM_TRAINING_RECORDS,
        help="number of train records",
    )
    # argparser.add_argument(
    #     "-v",
    #     "--num-validation-records",
    #     type=int,
    #     nargs="?",
    #     default=CFG.NUM_VALIDATION_RECORDS,
    #     help="number of validation records",
    # )
    argparser.add_argument(
        "-s",
        "--img-size",
        type=int,
        nargs="?",
        default=256,  # CFG.IMAGE_SIZE,
        help="image size",
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
    args = argparser.parse_args()

    # If statement to use multiprocessing if flag is dset
    if args.multiprocessing:
        logger.info("Writing TFRecords multiprocessing")
        write_mp_tfrecords(
            tfrecords_directory=Path(args.tfrecords_directory),
            img_directory=Path(args.img_directory),
            num_train_records=args.num_train_records,
            num_validation_records=args.num_validation_records,
            img_size=args.img_size,
        )
    else:
        logger.info("Writing TFRecords single process")
        write_sp_tfrecords(
            args.tfrecords_directory,
            args.img_directory,
            args.num_train_records,
            # args.num_validation_records,
            reshape_size=args.img_size,
        )
