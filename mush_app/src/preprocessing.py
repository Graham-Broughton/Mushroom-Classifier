import hashlib
import io
from os import environ
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from google.cloud import bigquery
from PIL import Image

PROJECT = environ.get("GCP_PROJECT_ID")


def get_model(path: Path):
    return tf.keras.models.load_model(path)


def get_image(url: str):
    """Retrieves an image from the specified URL and returns it as a PIL Image object.

    Args:
        url (str): The URL of the image to retrieve.

    Returns:
        PIL.Image.Image: The retrieved image.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure that the request was successful

    # Save the image to a BytesIO object (in-memory file)
    img_file = io.BytesIO(response.content)
    img = Image.open(img_file)

    return img


def hash_phone_number(phone_number):
    """Hashes a phone number using the SHA-256 algorithm.

    Args:
        phone_number (str): The phone number to be hashed.

    Returns:
        str: The hexadecimal representation of the hashed phone number.
    """
    # Create a new SHA-256 hash object
    hasher = hashlib.sha256()

    # Update the hash object with the phone number
    # Encode the phone number to bytes
    hasher.update(phone_number.encode("utf-8"))

    # Return the hexadecimal representation of the digest
    return hasher.hexdigest()


def initialize_database(
    dataset_name="image_database",
    table_name="image_data",
    location: str = "us-central1",
):
    """Initializes the database by creating the dataset and table in Google BigQuery if they don't exist.

    Args:
        dataset_name (str): The name of the dataset. Default is 'image_database'.
        table_name (str): The name of the table. Default is 'image_data'.
    """
    # Create the dataset if it doesn't exist
    client = bigquery.Client(project=str(PROJECT))
    dataset_ref = bigquery.DatasetReference(str(PROJECT), str(dataset_name))
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location
    dataset = client.create_dataset(dataset, exists_ok=True)

    # Create the table if it doesn't exist
    table_ref = dataset.table(table_name)
    schema = [
        bigquery.SchemaField("hash_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("image", "BYTES", mode="REQUIRED"),
    ]
    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table, exists_ok=True)


def insert_image_data(
    hash_id: str,
    img: Image,
    dataset_name="image_database",
    table_name="image_data",
    client=None,
):
    """Inserts image data into the database.

    Args:
        hash_id (str): The unique identifier for the image.
        img (PIL.Image): The image to be inserted into the database.
    """
    # Initialize the BigQuery client
    if not client:
        client = bigquery.Client(project=str(PROJECT))
    initialize_database(dataset_name, table_name)

    # Get the dataset and table references
    dataset_ref = bigquery.DatasetReference(str(PROJECT), str(dataset_name))
    table_ref = dataset_ref.table("image_data")

    # Convert PIL image to binary format
    img_byte_arr = io.BytesIO()
    img.save(
        img_byte_arr, format="JPEG"
    )  # Assuming format is JPEG, adjust if necessary
    img_blob = img_byte_arr.getvalue()

    # Create the row to be inserted
    row_to_insert = {
        "hash_id": hash_id,
        "image": img_blob,
    }

    # Insert the row into the table
    client.insert_rows(client.get_table(table_ref), [row_to_insert])


def process_image(url, hash_id, image_size):
    """Process an image by retrieving it from a given URL, inserting its data into a database using a hash ID,
    and resizing the image to a specified size.

    Args:
        url (str): The URL of the image.
        hash_id (str): The hash ID used for inserting the image data into the database.
        image_size (tuple): The desired size of the image after resizing.

    Returns:
        img (tf.Tensor): The processed image as a TensorFlow tensor.
    """
    img = get_image(url)
    insert_image_data(
        hash_id,
        img,
    )
    img = tf.image.resize(tf.convert_to_tensor(np.array(img)), image_size)
    return img


def load_dataset(url_list, phone_number, image_size):
    """Loads and preprocesses a dataset of images.

    Args:
        url_list (list): A list of image URLs.
        phone_number (str): The phone number associated with the dataset.
        image_size (tuple): The desired size of the images.

    Returns:
        ds (tf.data.Dataset): A TensorFlow dataset containing the preprocessed images.
    """
    hash_id = hash_phone_number(phone_number)
    imgs_list = list(
        map(
            lambda x: process_image(*x, image_size),
            zip(url_list, [hash_id] * len(url_list)),
        )
    )
    ds = tf.data.Dataset.from_tensor_slices(imgs_list)
    ds = ds.batch(1)
    return ds


if __name__ == "__main__":
    initialize_database()
