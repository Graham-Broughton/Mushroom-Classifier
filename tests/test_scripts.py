import unittest
from os import environ
from pathlib import Path

import scripts.delete_unused_images as delete_script  # Adjust the import

root = Path(environ.get("PYTHONPATH", "."))
data = root / "training" / "data"


class TestDeletedImages(unittest.TestCase):
    """Test the delete_unused_images.py script."""

    def test2018DeletedImages(self):
        # Setup a test environment if needed
        imagedir2018 = data / "raw" / "2018" / "train_val2018"

        # Call the script function
        delete_script.delete_2018_images(imagedir2018)

        # Add assertions to verify expected outcomes
        for p in imagedir2018.iterdir():
            if p.is_dir():
                if "Fungi" in p.name:
                    continue
                else:
                    self.assertFalse(p.exists())

    def test2021DeletedImages(self):
        # Setup a test environment if needed
        imagedir2021 = data / "raw" / "2021"

        # Call the script function
        delete_script.delete_2021_images(imagedir2021)

        # Add assertions to verify expected outcomes
        for p in ["train", "val"]:
            for d in (imagedir2021 / p).iterdir():
                if d.is_dir():
                    if "Fungi" in d.name:
                        continue
                    else:
                        self.assertFalse(d.exists())


if __name__ == "__main__":
    unittest.main()
