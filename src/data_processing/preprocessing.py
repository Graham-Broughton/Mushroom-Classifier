import json
import numpy as np
import pandas as pd
from loguru import logger
from prefect import flow, task
from tqdm import tqdm


@task
def class_priors(df: pd.DataFrame) -> np.ndarray:
    """Calculates the class priors for a given DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the class labels.

    Returns:
        class_priors (np.ndarray): An array containing the class priors.
    """
    class_priors = np.zeros(len(df["class_id"].unique()))
    for species in df["class_id"].unique():
        class_priors[species] = len(df[df["class_id"] == species])

    class_priors = class_priors / sum(class_priors)
    return class_priors


@task
def month_distributions(df):
    """Calculates the distribution of mushroom classes for each month in the dataset.

    Args:
        df (DataFrame): The input DataFrame containing the mushroom data.

    Returns:
        dict: A dictionary containing the distribution of mushroom classes for each month.
    """
    month_distributions = {}

    for _, observation in tqdm(df.iterrows(), total=len(df)):
        month = str(observation["date"].month)
        if month not in month_distributions:
            month_distributions[month] = np.zeros(len(df["class_id"].unique()))
        else:
            class_id = observation.class_id
            month_distributions[month][class_id] += 1

    for key, value in month_distributions.items():
        month_distributions[key] = value / sum(value)
    return month_distributions


@task
def parse_json(filepath, is_test=False, categories=None):
    """Parses a JSON file and returns relevant dataframes.

    Args:
        filepath (pathlib.Path): The path to the JSON file.
        is_test (bool, optional): Whether the JSON file is a test file. Defaults to False.
        categories (DataFrame, optional): A dataframe containing categories. Defaults to None.

    Returns:
        DataFrame: A dataframe containing information.
        DataFrame: A dataframe containing images.
        DataFrame: A dataframe containing annotations (if not a test file).
        DataFrame: A dataframe containing categories (if categories parameter is not None and not a test file).
    """
    with open(filepath, "r") as f:
        res = json.load(f)
    info = pd.DataFrame.from_dict(res["info"], orient="index")
    images = pd.DataFrame(res["images"]).set_index("id")
    if not is_test:
        annotations = pd.DataFrame(res["annotations"]).set_index("id")
        if categories:
            categories = pd.DataFrame(res["categories"]).set_index("id")
            return info, images, annotations, categories
        return info, images, annotations

    return info, images


@task
def join_dataframes(images, annotations, categories, set=None, locations=None):
    """Join dataframes containing information about images, annotations, categories, and locations (optional).
    Only categories with the supercategory 'Fungi' are included.

    Args:
        images (DataFrame): dataframe containing information about images
        annotations (DataFrame): dataframe containing information about annotations
        categories (DataFrame): dataframe containing information about categories
        locations (DataFrame, optional): dataframe containing information about image locations

    Returns:
        df (DataFrame): merged dataframe with selected columns dropped
    """
    categories = categories[categories["supercategory"] == "Fungi"].rename(
        columns={"id": "category_id"}
    )
    if locations is None:  # some datasets do not have location information
        df = pd.merge(
            categories, annotations, right_on="category_id", left_index=True
        ).merge(images, left_on="image_id", right_index=True)
    else:
        df = (
            pd.merge(categories, annotations, on="category_id")
            .merge(images, on="image_id")
            .merge(locations, on="image_id")
        )
    df = df.drop(
        ["supercategory", "kingdom", "image_id", "valid", "license", "rights_holder"],
        errors="ignore",
    )
    if set is not None:
        df["set"] = set
    return df


@flow(name='Parse2018Data')
def parse_2018_data(data_root):
    """Parses the 2018 mushroom dataset from the given data root directory.

    Args:
        data_root (pathlib.Path): The root directory of the dataset.

    Returns:
        DataFrame: A dataframe containing the parsed data.
    """
    logger.info(f"Parsing 2018 data from {data_root}")

    # Parse train and validation data
    timages2018, tanno2018, vimages2018, vanno2018 = [
        parse_json(data_root / f"{s}2018.json")[1:]
        for s in ["train", "val"]
    ]

    # Parse train and validation locations
    tloc, vloc = [
        pd.read_json(data_root / "inat2018_locations" / f"{s}2018_locations.json").set_index("id")
        for s in ["train", "val"]
    ]

    # Parse categories
    with open(data_root / "categories.json", "r") as f:
        cats = pd.DataFrame(json.load(f))

    # Join dataframes and save which set they are from
    val = join_dataframes(vimages2018, vanno2018, cats, locations=vloc, set="val")
    train = join_dataframes(timages2018, tanno2018, cats, locations=tloc, set="train")
    df = pd.concat([train, val]).reset_index(drop=True)
    df["dataset"] = "2018"

    # Create new directories and paths
    df["file_name"] = df["file_name"].str.split("/").str[-1]
    df["specific_epithet"] = df["name"].str.split().str[-1]
    df["image_dir_name"] = df[
        ["phylum", "class", "order", "family", "genus", "specific_epithet"]
    ].apply(lambda x: f"Fungi_{'_'.join(x)}", axis=1)

    # Drop unneeded columns and rename others
    df = df.drop(["category_id", "date_c"], axis=1).rename(
        columns={
            "lon": "longitude",
            "lat": "latitude",
            "loc_uncert": "location_uncertainty",
        }
    )
    logger.debug(f"df shape {df.shape}")
    return df


@flow(name='Parse2021Data')
def parse_2021_data(data_root):
    """Parses 2021 mushroom data from the given data root directory.

    Args:
        data_root (pathlib.Path): The root directory of the 2021 mushroom data.

    Returns:
        DataFrame: A concatenated dataframe of the parsed mushroom data.
    """
    logger.info(f"Parsing 2021 data from {data_root}")
    sets = ["train", "val", "train_mini"]

    dfs = [
        join_dataframes(
            *parse_json(data_root / f"{s}.json", categories=True)[:3], set=s
        )
        for s in sets
    ]
    df = pd.concat(dfs, ignore_index=True)

    df["dataset"] = "2021"
    df["file_name"] = df["file_name"].str.split("/").str[-1]
    df["image_dir_name"] = df["image_dir_name"].apply(
        lambda x: "_".join(x.split("_")[1:])
    )
    df = df.drop(["category_id", "common_name"], axis=1)
    logger.debug(f"df shape {df.shape}")
    return 


@flow(name='JoinDatasets')
def join_datasets(gcs_bucket: str, root: str) -> tuple:
    """Join two mushroom datasets, parse date column, create file path and GCS path columns,
    create class ID column, calculate month distribution and class prior, and return the
    concatenated dataframe and month distribution as a tuple.

    Args:
        gcs_bucket (str): The name of the Google Cloud Storage bucket.
        root (pathlib.Path): The root directory of the mushroom datasets.

    Returns:
        tuple: A tuple containing the concatenated dataframe and month distribution.
    """
    df1 = parse_2018_data(root)
    df2 = parse_2021_data(root)
    logger.info(f"Joining all datasets from {root}")
    df = pd.concat([df1, df2], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    df["file_path"] = (
        "Mushroom-Classifier/data/" + df["image_dir_name"] + "/" + df["file_name"]
    )
    df["gcs_path"] = (
        f"gs://{gcs_bucket}/train/" + df["image_dir_name"] + "/" + df["file_name"]
    )
    df["class_id"] = df["name"].astype("category").cat.codes

    month_distribution = month_distributions(df)
    class_prior = class_priors(df)

    df["class_priors"] = df["class_id"].map(dict(enumerate(class_prior)))

    logger.debug(f"df shape {df.shape}")
    return df, month_distribution


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    root = os.environ["ROOT"]
    data = root / "data"
    df, month_distribution, class_prior = join_datasets(os.environ["GCS_BUCKET"], data)
    df.to_csv(data / "train.csv", index=False)
