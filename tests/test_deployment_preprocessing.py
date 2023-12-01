import io
import unittest
from os import environ
from unittest.mock import MagicMock, patch

import tensorflow as tf
from google.cloud import bigquery
from PIL import Image

from mush_app.src.preprocessing import (
    get_image,
    hash_phone_number,
    initialize_database,
    insert_image_data,
    load_dataset,
    process_image,
)

test_urls = [
    "https://placebear.com/g/200/200",
    "https://via.placeholder.com/300.png/09f/fff",
]
PROJECT = environ.get("GCP_PROJECT_ID")


class TestPreprocessing(unittest.TestCase):
    def test_get_image(self):
        test_url = test_urls[0]
        result = get_image(test_url)
        self.assertIsInstance(result, Image.Image)

    def test_hash_phone_number(self):
        phone_number = "1234567890"
        result = hash_phone_number(phone_number)
        self.assertEqual(len(result), 64)  # SHA-256 hash is 64 characters long
        self.assertIsInstance(result, str)

    @patch("google.cloud.bigquery.Client")
    def test_initialize_database(self, mock_bigquery_client):
        # Mock bigquery client
        mock_client = mock_bigquery_client.return_value
        mock_client.create_dataset = MagicMock()
        mock_client.create_table = MagicMock()

        # Call the function
        initialize_database()

        # Assert that the bigquery Client was created with the correct project
        mock_bigquery_client.assert_called_with(project=str(PROJECT))

        # Assert that create_dataset was called
        mock_client.create_dataset.assert_called()

        # Assert that create_table was called
        mock_client.create_table.assert_called()


    @patch("mush_app.src.preprocessing.bigquery.Client")
    @patch("mush_app.src.preprocessing.bigquery.DatasetReference")
    def test_insert_image_data(self, mock_dataset_reference, mock_bigquery_client):
        # Setup mock client and other mock objects
        mock_client = mock_bigquery_client.return_value
        mock_table = MagicMock()

        # Mock the DatasetReference and its table method
        mock_dataset_ref = mock_dataset_reference.return_value
        mock_dataset_ref.table.return_value = mock_table

        # Call the function with mocked dependencies
        hash_id = "dummy_hash_id"
        img = Image.new("RGB", (100, 100), color="red")
        insert_image_data(hash_id, img, "image_database", "image_data", mock_client)

        # Assertions
        mock_dataset_reference.assert_called_with(PROJECT, "image_database")
        mock_dataset_ref.table.assert_called_with("image_data")

        # Ensure the image is converted to binary format
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_blob = img_byte_arr.getvalue()
        expected_row = {
            "hash_id": hash_id,
            "image": img_blob,
        }
        mock_client.insert_rows_json.assert_called_with(mock_table, [expected_row])

    @patch("mush_app.src.preprocessing.get_image")
    @patch("mush_app.src.preprocessing.insert_image_data")
    def test_process_image(self, mock_insert_image_data, mock_get_image):
        # Setup the mocks
        test_url = test_urls[0]
        test_hash_id = "1234567890"
        test_image_size = (64, 64)

        # Create a mock image
        mock_image = Image.new("RGB", (100, 100), color="red")
        mock_get_image.return_value = mock_image

        # Call the function
        processed_image = process_image(test_url, test_hash_id, test_image_size)

        # Assertions
        mock_get_image.assert_called_with(test_url)
        mock_insert_image_data.assert_called_with(test_hash_id, mock_image)

        # Check if the image is resized correctly
        self.assertEqual(processed_image.shape, (64, 64, 3))

    def test_load_dataset(self):
        # Define test data
        url_list = test_urls
        phone_number = "1234567890"
        image_size = (64, 64)

        # Mock the necessary functions
        mock_hash_phone_number = MagicMock(return_value="dummy_hash_id")
        mock_process_image = MagicMock(
            return_value=tf.ones((*image_size, 3), dtype=tf.float32)
        )  # Mock the processed image

        # Patch the necessary functions
        with patch(
            "mush_app.src.preprocessing.hash_phone_number", mock_hash_phone_number
        ), patch("mush_app.src.preprocessing.process_image", mock_process_image):
            # Call the function
            ds = load_dataset(url_list, phone_number, image_size)

            # Assertions
            self.assertEqual(
                len(ds), len(url_list)
            )  # Check if the dataset has the same number of elements as the URL list
            self.assertTrue(
                isinstance(ds, tf.data.Dataset)
            )  # Check if the dataset is a TensorFlow dataset
            self.assertEqual(
                ds.element_spec,
                (tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32)),
            )  # Check the element spec of the dataset

            # Check if the necessary functions were called with the correct arguments
            mock_hash_phone_number.assert_called_once_with(phone_number)
            mock_process_image.assert_called_with(
                url_list[-1], "dummy_hash_id", image_size
            )


if __name__ == "__main__":
    unittest.main()
