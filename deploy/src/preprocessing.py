import hashlib
import io
import sqlite3

import numpy as np
import requests
import tensorflow as tf
from PIL import Image


def get_image_and_exif(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure that the request was successful

    # Save the image to a BytesIO object (in-memory file)
    img_file = io.BytesIO(response.content)
    img = Image.open(img_file)

    return img

def hash_phone_number(phone_number):
    # Create a new SHA-256 hash object
    hasher = hashlib.sha256()

    # Update the hash object with the phone number
    # Encode the phone number to bytes
    hasher.update(phone_number.encode("utf-8"))

    # Return the hexadecimal representation of the digest
    return hasher.hexdigest()

def initialize_database():
    conn = sqlite3.connect("image_data.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS image_data (
            id INTEGER PRIMARY KEY,
            hash_id TEXT,
            image BLOB,
        )
    """
    )
    conn.commit()
    conn.close()

def insert_image_data(
    hash_id: str,
    img: Image,
):
    # Convert PIL image to binary format
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img.format)
    img_blob = img_byte_arr.getvalue()

    # Connect to the database
    conn = sqlite3.connect("image_data.db")
    cursor = conn.cursor()

    # Insert the data
    cursor.execute(
        """
        INSERT INTO image_data (hash_id, image)
        VALUES (?, ?)
    """,
        (hash_id, img_blob),
    )

    conn.commit()
    conn.close()

def process_image(url, hash_id, image_size):
    img = get_image_and_exif(url)
    insert_image_data(
        "images.db",
        hash_id,
        img,
    )
    img = tf.image.resize(tf.convert_to_tensor(np.array(img)), image_size)
    return img

def load_dataset(url_list, phone_number, image_size):
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
