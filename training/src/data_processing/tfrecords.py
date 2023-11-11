import warnings
from os import environ
from pathlib import Path

import cv2
import pandas as pd
from loguru import logger
from tqdm import trange
from prefect import task, Flow

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from training.train_config import GCFG

warnings.filterwarnings("ignore")

AUTO = tf.data.experimental.AUTOTUNE


# @task
def load_dataframe(root_path):
    """This function loads a dataframe from the given root path.

    Parameters:
        root_path (pathlib.Path): The root path where the dataframe is located.

    Returns:
        DataFrame: The loaded dataframe.
    """
    df = pd.read_csv(root_path.parent / "train.csv")
    val = df[df["dset"] == "val"].sample(frac=1, random_state=42).reset_index(drop=True)
    train = (
        df[df["dset"] == "train"].sample(frac=1, random_state=42).reset_index(drop=True)
    )
    del df
    logger.info("Loaded train and val dataframes")
    logger.debug(f"Train shape: {train.shape}  :  val shape: {val.shape}")
    return train, val


# @task
def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# @task
def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# @task
def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# @task
def serialize_example(
    feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8
):
    """This function serializes an example with given features.

    Returns:
        str: Serialized example.
    """
    feature = {
        "image/encoded": bytes_feature(feature0),
        "image/id": bytes_feature(feature1),
        "image/meta/dataset": int64_feature(feature2),
        "image/meta/longitude": float_feature(feature3),
        "image/meta/latitude": float_feature(feature4),
        "image/meta/date_encoded": float_feature(feature5),
        "image/meta/class_priors": float_feature(feature6),
        "image/class/label": int64_feature(feature7),
        "image/class/text": bytes_feature(feature8),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# @task
def get_data(df, num_records=107):
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


# @Flow
def write_sp_tfrecords(tfrec_path, img_path, num_train_records, num_val_records, reshape_size):
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
    train, val = load_dataframe(root_path=img_path)
    for df, dset in zip([train, val], ["train", "val"]):
        num_records = num_train_records if dset == "train" else num_val_records
        IMGS, SIZE, CT = get_data(df, num_records)

        # iterate over the number of tfrecords
        for j in trange(CT):
            logger.info(f"Writing {j:02d} of {CT} {dset} tfrecords")
            CT2 = min(SIZE, len(IMGS) - j * SIZE)  # get the number of images in a tfrecord

            # create the path to write the tfrecord to
            path = tfrec_path / f"tfrecords-jpeg-{reshape_size}x{reshape_size}"
            path.mkdir(parents=True, exist_ok=True)

            with tf.io.TFRecordWriter(str(path / f"{dset}{j:02d}-{CT2}.tfrec")) as writer:
                for k in trange(CT2, leave=False):  # for each image in the tfrecord
                    if k % 100 == 0:
                        logger.info(f"Writing {k:02d} of {CT2} {dset} tfrecord images")

                    # load image from disk, change RGB to cv2 default BGR format, resize to reshape_sizes and encode as jpeg
                    img = cv2.imread(str(img_path / f"{IMGS[SIZE * j + k]}"))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, (reshape_size, reshape_size), cv2.INTER_CUBIC)
                    img = cv2.imencode(".jpg", img)[1].tobytes()

                    # read specific row from dataframe to get data and serialize it
                    row = df.loc[df.file_path == IMGS[SIZE * j + k]].iloc[0]
                    example = serialize_example(
                        img,
                        row.file_name.split('.')[0],
                        row.dataset,
                        row.longitude,
                        row.latitude,
                        row.norm_date,
                        row.class_priors,
                        row.class_id,
                        f"{row.genus}_{row.special_epithet}",
                    )
                    writer.write(example)


# @task
def get_mp_params(tfrec_path, img_path, num_records, reshape_size, dset, df):
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
    IMGS, SIZE, CT = get_data(df, num_records)
    param_list = []  # multiprocessing requires the arguments to be in a list
    for j in range(CT):  # for each tfrecord
        CT2 = min(
            SIZE, len(IMGS) - j * SIZE
        )  # find the number of images in the tfrecord

        # create the path to write the tfrecord to
        path = tfrec_path / f"tfrecords-jpeg-{reshape_size}x{reshape_size}"
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        image_paths = []
        image_filenames = []

        for k in range(CT2):  # for each image in the tfrecord
            # get the image path and filename and save to lists
            img_filename = IMGS[SIZE * j + k]
            image_filenames.append(img_filename)
            image_path = str(img_path / img_filename)
            image_paths.append(image_path)

        # once the lists are saved, create a dictionary of parameters and append to the param list
        params = {
            "df": df,
            "path": str(path / f"{dset}{j:02d}-{CT2}.tfrec"),
            "reshape_size": reshape_size,
            "image_paths": image_paths,
            "image_filenames": image_filenames,
        }
        param_list.append(params)
    return param_list


# @task
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
    image_paths = params["image_paths"]
    image_filenames = params["image_filenames"]

    with tf.io.TFRecordWriter(path) as writer:
        # unnecessary to use CT2 since the paths and filenames were derived from it
        for image_path, image_filename in zip(
            image_paths, image_filenames
        ):  # for each image in the tfrecord..
            # load image from disk, change RGB to cv2 default BGR format, resize to reshape_sizes and encode as jpeg
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(
                img, (reshape_size, reshape_size), cv2.INTER_CUBIC
            )  # Fix incorrect colors
            img = cv2.imencode(".jpg", img)[1].tobytes()

            # read specific row from dataframe to get data and serialize it
            row = df.loc[df.file_path == image_filename]
            example = serialize_example(
                img,
                row.filename.values[0].split('.')[0],
                row.dataset.values[0],
                row.longitude.values[0],
                row.latitude.values[0],
                row.norm_date.values[0],
                row.class_priors.values[0],
                row.class_id.values[0],
                f"{row.genus}_{row.special_epithet}"
            )
            writer.write(example)


# @Flow
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
    from tqdm.contrib.concurrent import process_map

    train, val = load_dataframe(root_path=kwargs["img_directory"])
    for df, dset in zip(
        [train, val], ["train", "val"]
    ):  # for each dataframe get the parameters for multiprocessing
        params = get_mp_params(
            kwargs["tfrecords_directory"],
            kwargs["img_directory"],
            kwargs["num_train_records"]
            if dset == "train"
            else kwargs["num_validation_records"],
            kwargs["img_size"],
            dset,
            df,
        )
        logger.info(f"Starting multiprocessing for {dset}")
        # while still in the loop, use multiprocessing to write the tfrecords
        process_map(write_single_mp_tfrecord, params, max_workers=8)
        logger.info(f"Finished multiprocessing for {dset}")


if __name__ == "__main__":
    import argparse  # only needed when called as script

    CFG = GCFG()  # need to initialize to make the dataclass work properly

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    # argparse to allow for command line arguments
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
        default=CFG.NUM_TRAINING_RECORDS,
        help="number of train records",
    )
    argparser.add_argument(
        "-v", "--num-validation-records",
        type=int,
        nargs="?",
        default=CFG.NUM_VALIDATION_RECORDS,
        help="number of validation records",
    )
    argparser.add_argument(
        "-s", "--img-size", 
        type=int, 
        nargs="?", 
        default=CFG.IMAGE_SIZE, 
        help="image size"
    )
    argparser.add_argument(
        "-m", "--multiprocessing",
        type=str2bool,
        nargs='?',
        const=False,
        default=True,
        # action="store_true",
        help="whether to use multiprocessing",
    )
    args = argparser.parse_args()
    print(args)
    # If statement to use multiprocessing if flag is dset
    if not args.multiprocessing:
        logger.info("Writing TFRecords multiprocessing")
        print(args)
        write_mp_tfrecords(
            tfrecords_directory=Path(args.tfrecords_directory),
            img_directory=Path('./training/data/raw'), #Path(args.img_directory),
            num_train_records=100, #args.num_train_records,
            num_validation_records=args.num_validation_records,
            img_size=args.img_size,
        )
    else:
        logger.info("Writing TFRecords single process")
        write_sp_tfrecords(
            args.tfrecords_directory,
            args.img_directory,
            args.num_train_records,
            args.num_validation_records,
            reshape_size=args.img_size,
        )
