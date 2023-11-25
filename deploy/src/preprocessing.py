import hashlib
import io
import re
import sqlite3
from datetime import datetime

import exifread
import numpy as np
import requests
import tensorflow as tf
from PIL import Image


def get_image_and_exif(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure that the request was successful

    # Save the image to a BytesIO object (in-memory file)
    img_file = io.BytesIO(response.content)

    # Use exifread to extract EXIF data
    tags = exifread.process_file(img_file)

    # For further processing, load the image using PIL
    img_file.seek(0)  # Reset file pointer to the beginning
    img = Image.open(img_file)

    return img, tags


def get_image_dimensions(exif_data):
    # The tags can have different names depending on the camera and the file format
    width_tag = exif_data.get("EXIF ExifImageWidth") or exif_data.get(
        "Image ImageWidth"
    )
    height_tag = exif_data.get("EXIF ExifImageLength") or exif_data.get(
        "Image ImageLength"
    )

    if width_tag and height_tag:
        width = width_tag.values[0]
        height = height_tag.values[0]
        return width, height
    else:
        return None, None


def hash_phone_number(phone_number):
    # Create a new SHA-256 hash object
    hasher = hashlib.sha256()

    # Update the hash object with the phone number
    # Encode the phone number to bytes
    hasher.update(phone_number.encode("utf-8"))

    # Return the hexadecimal representation of the digest
    return hasher.hexdigest()


def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ["S", "W"]:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return degrees + minutes + seconds


def get_gps_coords(exif_data):
    lat = None
    lon = None

    gps_latitude = exif_data.get("GPS GPSLatitude")
    gps_latitude_ref = exif_data.get("GPS GPSLatitudeRef")
    gps_longitude = exif_data.get("GPS GPSLongitude")
    gps_longitude_ref = exif_data.get("GPS GPSLongitudeRef")

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = get_decimal_from_dms(gps_latitude.values, gps_latitude_ref.values)
        lon = get_decimal_from_dms(gps_longitude.values, gps_longitude_ref.values)

    return lat, lon


def get_date_and_gps_coords(exif_data):
    date_taken_str = str(exif_data.get("EXIF DateTimeOriginal"))
    gps_coords = get_gps_coords(
        exif_data
    )  # Assuming this function returns the GPS coordinates correctly

    # Parse the date string and convert it to a datetime object
    if date_taken_str:
        # Ensure the date string is in the expected format
        matched = re.match(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", date_taken_str)
        if matched:
            date_taken = datetime.strptime(matched.group(), "%Y:%m:%d %H:%M:%S")
            day_of_year = date_taken.timetuple().tm_yday
            month = date_taken.month
            return day_of_year, month, gps_coords
        else:
            # Handle unexpected date format
            print("Date format is not as expected.")
            return None, None, gps_coords
    else:
        # Handle cases where date is not present
        print("No date information found.")
        return None, None, gps_coords


def initialize_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS image_data (
            id INTEGER PRIMARY KEY,
            hash_id TEXT,
            image BLOB,
            height INTEGER,
            width INTEGER,
            latitude REAL,
            longitude REAL,
            day_of_year INTEGER,
            month INTEGER
        )
    """
    )
    conn.commit()
    conn.close()


def insert_image_data(
    db_name: str,
    hash_id: str,
    img: Image,
    height: int,
    width: int,
    latitude: float,
    longitude: float,
    day_of_year: int,
    month: int,
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
        INSERT INTO image_data (hash_id, image, height, width, latitude, longitude, day_of_year, month)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (hash_id, img_blob, height, width, latitude, longitude, day_of_year, month),
    )

    conn.commit()
    conn.close()


def get_exif_data(exif_data):
    width, height = get_image_dimensions(exif_data)
    day_of_year, month, gps_coords = get_date_and_gps_coords(exif_data)

    return width, height, day_of_year, month, gps_coords


def process_image(url, hash_id, image_size):
    img, exif_data = get_image_and_exif(url)
    width, height, day_of_year, month, gps_coords = get_exif_data(exif_data)
    insert_image_data(
        "images.db",
        hash_id,
        img,
        height,
        width,
        gps_coords[0],
        gps_coords[1],
        day_of_year,
        month,
    )
    img = tf.image.resize(tf.convert_to_tensor(np.array(img)), image_size)
    return img, gps_coords[0], gps_coords[1], day_of_year, month


def get_date_feats(meta):
    days = meta[2]
    months = meta[3]

    months_in_year = 12
    days_in_year = 365

    day_sin = tf.math.sin(days * (2 * np.pi / days_in_year))
    day_cos = tf.math.cos(days * (2 * np.pi / days_in_year))

    month_sin = tf.math.sin(months * (2 * np.pi / months_in_year))
    month_cos = tf.math.cos(months * (2 * np.pi / months_in_year))
    return tf.stack([day_sin, day_cos, month_sin, month_cos], axis=-1)


def map_coordinates_to_grid(meta, grid_size=(100, 100), normalize=True):
    latitudes, longitudes = meta[0], meta[1]

    normalized_lat = (latitudes + 90) / 180
    normalized_lon = (longitudes + 180) / 360

    grid_x = tf.cast(normalized_lon * grid_size[1], tf.float32)
    grid_y = tf.cast(normalized_lat * grid_size[0], tf.float32)

    grid_x = tf.clip_by_value(grid_x, 0, grid_size[1] - 1)
    grid_y = tf.clip_by_value(grid_y, 0, grid_size[0] - 1)

    if normalize:
        # Normalize to 0-1 range
        grid_x = grid_x / (grid_size[1] - 1)
        grid_y = grid_y / (grid_size[0] - 1)

    return tf.stack([grid_x, grid_y], axis=-1)


@tf.function
def process_meta(meta):
    date_feats = get_date_feats(meta)
    gps_feat = map_coordinates_to_grid(meta)
    meta = tf.concat([date_feats, gps_feat], axis=-1)
    return meta


def load_dataset(url_list, phone_number, image_size):
    hash_id = hash_phone_number(phone_number)
    imgs_exif_list = list(
        map(
            lambda x: process_image(*x, image_size),
            zip(url_list, [hash_id] * len(url_list)),
        )
    )
    img = tf.data.Dataset.from_tensor_slices([x[0] for x in imgs_exif_list])
    meta = tf.data.Dataset.from_tensor_slices([x[1:] for x in imgs_exif_list])
    ds = tf.data.Dataset.zip((img, meta))
    ds = ds.map(
        lambda x, y: (x, process_meta(y)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.batch(1)
    return ds
