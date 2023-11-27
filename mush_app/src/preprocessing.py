import hashlib
import io
import sqlite3

import numpy as np
import requests
import tensorflow as tf
from PIL import Image


def get_model(path):
    return tf.keras.models.load_model(path)


def get_image(url):
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


def initialize_database(database_name="image_data.db"):
    """Initializes the database by creating the 'image_data' table if it doesn't exist.

    Args:
        database_name (str): The name of the database file. Default is 'image_data.db'.
    """
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS image_data (
    id INTEGER PRIMARY KEY,
    hash_id TEXT,
    image BLOB)
"""
)
    conn.commit()
    conn.close()


def insert_image_data(hash_id: str, img: Image):
    """Inserts image data into the database.

    Args:
        hash_id (str): The unique identifier for the image.
        img (PIL.Image): The image to be inserted into the database.
    """
    initialize_database()
    # Convert PIL image to binary format
    img_byte_arr = io.BytesIO()
    img.save(
        img_byte_arr, format="JPEG"
    )  # Assuming format is JPEG, adjust if necessary
    img_blob = img_byte_arr.getvalue()

    # Connect to the database
    conn = sqlite3.connect("image_data.db")
    cursor = conn.cursor()

    # Insert the data using parameter substitution
    cursor.execute(
        "INSERT INTO image_data (hash_id, image) VALUES (?, ?)", (hash_id, img_blob)
    )

    conn.commit()
    conn.close()


def process_image(url, hash_id, image_size):
    """Process an image by retrieving it from a given URL, inserting its data into a database using a hash ID,
    and resizing the image to a specified size.

    Args:
        url (str): The URL of the image.
        hash_id (str): The hash ID used for inserting the image data into the database.
        image_size (tuple): The desired size of the image after resizing.

    Returns:
        tf.Tensor: The processed image as a TensorFlow tensor.
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
        tf.data.Dataset: A TensorFlow dataset containing the preprocessed images.
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
