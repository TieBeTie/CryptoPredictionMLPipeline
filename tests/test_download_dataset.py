#!/usr/bin/env python3
"""Test the download_dataset script."""
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

import src.data.download_dataset as download_script


class TestDownloadDataset(unittest.TestCase):
    """Test suite for the dataset download script."""

    @patch("src.data.download_dataset.kaggle")
    def test_download_dataset_calls_kaggle_api(self, mock_kaggle):
        """Test that download_dataset calls the Kaggle API correctly."""
        # Set up mock environment
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"}):
            # Mock the glob results
            with patch.object(Path, "glob") as mock_glob:
                mock_glob.return_value = [Path("data/raw/test.csv")]
                
                # Mock the file.stat() method
                mock_stat = MagicMock()
                mock_stat.st_size = 1024 * 10  # 10KB
                
                with patch("pathlib.Path.stat", return_value=mock_stat):
                    # Call the function under test
                    result = download_script.download_dataset()

                    # Assert Kaggle API was called
                    mock_kaggle.api.authenticate.assert_called_once()
                    mock_kaggle.api.dataset_download_files.assert_called_once_with(
                        dataset=download_script.DATASET_NAME,
                        path=download_script.RAW_DATA_PATH,
                        unzip=True,
                    )
                    
                    # Verify return value
                    self.assertEqual(len(result), 1)
                    self.assertEqual(result[0], Path("data/raw/test.csv"))

    def test_raw_directory_exists(self):
        """Test that the raw data directory exists or gets created."""
        download_script.create_directories()
        raw_path = Path("data/raw")
        self.assertTrue(raw_path.exists(), "Raw data directory does not exist")
        self.assertTrue(raw_path.is_dir(), "Raw data path is not a directory")

    @patch("src.data.download_dataset.download_dataset")
    @patch("src.data.download_dataset.track_with_dvc")
    def test_main_function_calls_download(self, mock_track_with_dvc, mock_download):
        """Test that the main function calls download_dataset and track_with_dvc."""
        # Configure mocks
        mock_download.return_value = [Path("data/raw/test.csv")]
        mock_track_with_dvc.return_value = True
        
        # Call function under test
        result = download_script.main()
        
        # Verify function calls and result
        mock_download.assert_called_once()
        mock_track_with_dvc.assert_called_once()
        self.assertTrue(result)

    @patch.dict("os.environ", clear=True)
    def test_missing_credentials_raises_error(self):
        """Test that missing Kaggle credentials raise an error."""
        with self.assertRaises(EnvironmentError):
            download_script.download_dataset()

    @patch("src.data.download_dataset.download_dataset")
    @patch("src.data.download_dataset.track_with_dvc")
    def test_main_function_handles_exception(self, mock_track_with_dvc, mock_download):
        """Test that main function handles exceptions correctly."""
        # Configure mock to raise exception
        mock_download.side_effect = Exception("Test error")
        
        # Call function under test
        result = download_script.main()
        
        # Verify result
        self.assertFalse(result)
        # track_with_dvc should not be called if download fails
        mock_track_with_dvc.assert_not_called()

    @patch("src.data.download_dataset.add_to_dvc")
    @patch("src.data.download_dataset.check_dvc_status")
    def test_track_with_dvc_success(self, mock_check_dvc_status, mock_add_to_dvc):
        """Test track_with_dvc success path."""
        # Configure mocks
        mock_add_to_dvc.return_value = True
        mock_check_dvc_status.return_value = True
        
        # Call function under test
        result = download_script.track_with_dvc()
        
        # Verify function calls and result
        mock_add_to_dvc.assert_called_once_with(download_script.RAW_DATA_PATH)
        mock_check_dvc_status.assert_called_once()
        self.assertTrue(result)

    @patch("src.data.download_dataset.add_to_dvc")
    @patch("src.data.download_dataset.check_dvc_status")
    def test_track_with_dvc_status_not_clean(self, mock_check_dvc_status, mock_add_to_dvc):
        """Test track_with_dvc when DVC status is not clean."""
        # Configure mocks
        mock_add_to_dvc.return_value = True
        mock_check_dvc_status.return_value = False
        
        # Call function under test
        result = download_script.track_with_dvc()
        
        # Verify function calls and result
        mock_add_to_dvc.assert_called_once_with(download_script.RAW_DATA_PATH)
        mock_check_dvc_status.assert_called_once()
        self.assertFalse(result)

    @patch("src.data.download_dataset.add_to_dvc")
    def test_track_with_dvc_exception(self, mock_add_to_dvc):
        """Test track_with_dvc exception handling."""
        # Configure mock to raise exception
        mock_add_to_dvc.side_effect = Exception("Test error")
        
        # Call function under test
        result = download_script.track_with_dvc()
        
        # Verify result
        self.assertFalse(result)

    @patch("src.data.download_dataset.kaggle")
    @patch("src.data.download_dataset.logger")
    def test_download_dataset_file_logging(self, mock_logger, mock_kaggle):
        """Test that download_dataset logs info about each downloaded file."""
        # Set up mock environment
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"}):
            # Test with multiple files
            file_list = [Path("data/raw/test1.csv"), Path("data/raw/test2.csv")]
            
            # Mock the glob and file stat results
            with patch.object(Path, "glob") as mock_glob:
                mock_glob.return_value = file_list
                
                # Mock the file.stat() method
                mock_stat = MagicMock()
                mock_stat.st_size = 1024 * 10  # 10KB
                
                with patch("pathlib.Path.stat", return_value=mock_stat):
                    # Call the function under test
                    download_script.download_dataset()
                    
                    # Check that logger.info was called for each file (at least 4 times)
                    # 1: Downloading dataset, 2-3: Found file logs, 4: Downloaded files summary
                    self.assertGreaterEqual(mock_logger.info.call_count, 4)

    # New test cases:

    @patch("src.data.download_dataset.kaggle")
    def test_empty_result_handling(self, mock_kaggle):
        """Test handling when no files are downloaded."""
        # Set up mock environment
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"}):
            # Mock an empty download result
            with patch.object(Path, "glob") as mock_glob:
                mock_glob.return_value = []
                
                # Call the function under test
                result = download_script.download_dataset()
                
                # Verify result is an empty list
                self.assertEqual(result, [])
                self.assertEqual(len(result), 0)

    @patch("src.data.download_dataset.kaggle")
    def test_handle_kaggle_api_error(self, mock_kaggle):
        """Test handling when Kaggle API raises an error."""
        # Set up mock environment
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"}):
            # Configure mock to raise exception
            mock_kaggle.api.dataset_download_files.side_effect = Exception("API Error")
            
            # Verify exception is raised
            with self.assertRaises(Exception):
                download_script.download_dataset()

    @patch("src.data.download_dataset.kaggle")
    def test_handle_malformed_csv(self, mock_kaggle):
        """Test handling when downloaded CSV files are malformed."""
        # Set up mock environment
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"}):
            # Create a test file path
            test_file = Path("data/raw/malformed.csv")
            
            # Mock the glob results
            with patch.object(Path, "glob") as mock_glob:
                mock_glob.return_value = [test_file]
                
                # Mock the file.stat() method
                mock_stat = MagicMock()
                mock_stat.st_size = 0  # Zero-sized file
                
                with patch("pathlib.Path.stat", return_value=mock_stat):
                    # Mock Path.exists to return True
                    with patch.object(Path, "exists") as mock_exists:
                        mock_exists.return_value = True
                        
                        # Call the function under test
                        result = download_script.download_dataset()
                        
                        # Verify result
                        self.assertEqual(len(result), 1)
                        # Size is checked in the download function
                        self.assertEqual(result[0].stat().st_size, 0)

    @given(
        file_names=st.lists(
            st.text(
                alphabet=st.characters(blacklist_categories=('Cs',)),
                min_size=1,
                max_size=20
            ).map(lambda s: s + ".csv"),
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=50)
    @patch("src.data.download_dataset.kaggle")
    def test_property_based_file_handling(self, mock_kaggle, file_names):
        """Property-based test for file handling with various file names."""
        # Set up mock environment
        with patch.dict("os.environ", {"KAGGLE_USERNAME": "test", "KAGGLE_KEY": "test"}):
            # Create Path objects from file names
            file_paths = [Path("data/raw") / name for name in file_names]
            
            # Mock the glob results
            with patch.object(Path, "glob") as mock_glob:
                mock_glob.return_value = file_paths
                
                # Mock the file.stat() method
                mock_stat = MagicMock()
                mock_stat.st_size = 1024 * 10  # 10KB
                
                with patch("pathlib.Path.stat", return_value=mock_stat):
                    # Call the function under test
                    result = download_script.download_dataset()
                    
                    # Verify result
                    self.assertEqual(len(result), len(file_names))
                    for actual, expected in zip(result, file_paths):
                        self.assertEqual(actual, expected)


# Moved benchmark test out of the class to use pytest's fixture properly
@pytest.mark.benchmark
def test_create_directories_performance(benchmark):
    """Test the performance of create_directories function."""
    # Use benchmark to measure the function's performance
    benchmark(download_script.create_directories)


@patch("src.data.download_dataset.download_dataset")
@patch("src.data.download_dataset.track_with_dvc")
def test_return_value_of_main(mock_track_with_dvc, mock_download):
    """Test the return value of main() when it completes successfully."""
    # Configure mocks for success case
    mock_download.return_value = [Path("data/raw/test.csv")]
    mock_track_with_dvc.return_value = True
    
    # Call function and check return value
    result = download_script.main()
    assert result is True
    
    # Call again with track_with_dvc returning False (should still return True)
    mock_track_with_dvc.return_value = False
    result = download_script.main()
    assert result is True


if __name__ == "__main__":
    unittest.main()
