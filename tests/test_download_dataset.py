#!/usr/bin/env python3
"""Test the download_dataset script."""
import unittest
from pathlib import Path
from unittest.mock import patch

import src.data.download_dataset as download_script


class TestDownloadDataset(unittest.TestCase):
    """Test suite for the dataset download script."""

    @patch("src.data.download_dataset.kaggle")
    def test_download_dataset_calls_kaggle_api(self, mock_kaggle):
        """Test that download_dataset calls the Kaggle API correctly."""
        # Set up mock environment
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"}):
            # Call the function under test
            download_script.download_dataset()

            # Assert Kaggle API was called
            mock_kaggle.api.authenticate.assert_called_once()
            mock_kaggle.api.dataset_download_files.assert_called_once_with(
                dataset=download_script.DATASET_NAME,
                path=download_script.RAW_DATA_PATH,
                unzip=True,
            )

    def test_raw_directory_exists(self):
        """Test that the raw data directory exists or gets created."""
        download_script.create_directories()
        raw_path = Path("data/raw")
        self.assertTrue(raw_path.exists(), "Raw data directory does not exist")
        self.assertTrue(raw_path.is_dir(), "Raw data path is not a directory")

    @patch("src.data.download_dataset.download_dataset")
    def test_main_function_calls_download(self, mock_download):
        """Test that the main function calls download_dataset."""
        download_script.main()
        mock_download.assert_called_once()

    @patch.dict("os.environ", clear=True)
    def test_missing_credentials_raises_error(self):
        """Test that missing Kaggle credentials raise an error."""
        with self.assertRaises(EnvironmentError):
            download_script.download_dataset()


if __name__ == "__main__":
    unittest.main()
