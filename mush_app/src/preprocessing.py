import hashlib
import re
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import tensorflow as tf
from google.cloud import bigquery
import google.auth
from PIL import Image
from urllib.parse import urlparse
import logging

credentials, project = google.auth.default()


def get_model(path: Path):
    try:
        model = tf.keras.models.load_model(path)
        logging.info(f"Model loaded successfully from {path}")
        return model
    except IOError:
        logging.error(f"Failed to load model from {path}. File not found or inaccessible.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        return None


def is_valid_url(url: str) -> bool:
    """Check if the URL is valid and well-formed."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_image(url: str):
    """Retrieves an image from the specified URL and returns it as a PIL Image object.
    
    Args:
        url (str): The URL of the image to retrieve.

    Returns:
        PIL.Image.Image: The retrieved image, or None if an error occurs.
    """
    if not is_valid_url(url):
        logging.warning(f"Invalid URL provided: {url}")
        return None

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img_file = io.BytesIO(response.content)
        img = Image.open(img_file)
        # Optional: Check image properties here (size, format, etc.)
        return img
    except requests.RequestException as e:
        logging.error(f"Network error occurred while retrieving image: {e}")
        return None
    except IOError:
        logging.error("Invalid image data received.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while processing the image: {e}")
        return None


def is_valid_phone_number(phone_number: str) -> bool:
    """Validate the phone number format. Adjust the regex according to the expected format."""
    pattern = re.compile(r"^\+\d{1,3}\s?\d{4,14}$")  # Example pattern for international numbers
    return pattern.match(phone_number) is not None

def hash_phone_number(phone_number: str, salt: str):
    """Hashes a phone number using the SHA-256 algorithm, with added salt for security.

    Args:
        phone_number (str): The phone number to hash.
        salt (str): A salt value for additional security.

    Returns:
        str: The hashed phone number.
    """
    if not is_valid_phone_number(phone_number):
        logging.warning(f"Invalid phone number format: {phone_number}")
        return None

    salted_number = phone_number + salt
    return hashlib.sha256(salted_number.encode()).hexdigest()


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
        client = bigquery.Client(project=project, credentials=credentials)
    
    # Get the dataset and table references
    dataset_ref = bigquery.DatasetReference(project, str(dataset_name))

    def initialize_database(
        location: str = "us-central1",
    ):
        """Initializes the database by creating the dataset and table in Google BigQuery if they don't exist.

        Args:
            dataset_name (str): The name of the dataset. Default is 'image_database'.
            table_name (str): The name of the table. Default is 'image_data'.
        """
        # Create the dataset if it doesn't exist
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset = client.create_dataset(dataset, exists_ok=True)

        # Create the table if it doesn't exist
        table_ref = dataset.table(table_name)
        schema = [
            bigquery.SchemaField("timestamp", "DATETIME", mode="REQUIRED"),
            bigquery.SchemaField("hash_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("image", "BYTES", mode="REQUIRED", default_value_expression="CURRENT_DATETIME"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table, exists_ok=True)
     
    initialize_database()

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
