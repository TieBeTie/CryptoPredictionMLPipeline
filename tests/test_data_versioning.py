#!/usr/bin/env python3
"""Test data versioning with DVC."""
import unittest
from pathlib import Path

from src.data.dvc_utils import check_dvc_status


class TestDataVersioning(unittest.TestCase):
    """Test suite for data versioning."""

    def test_raw_data_exists(self):
        """Test that raw data files exist."""
        raw_path = Path("data/raw/test")
        self.assertTrue(raw_path.exists(), "Raw data directory does not exist")

        # Check for at least one CSV file
        csv_files = list(raw_path.glob("*.csv"))
        self.assertGreater(len(csv_files), 0, "No CSV files found in raw data directory")

        # Check that files have content
        for file in csv_files:
            self.assertGreater(file.stat().st_size, 0, f"File {file} is empty")

    def test_dvc_status_clean(self):
        """Test that DVC status is clean."""
        is_clean = check_dvc_status()
        self.assertTrue(is_clean, "DVC status is not clean")

    def test_dvc_lock_exists(self):
        """Test that DVC lock file exists."""
        dvc_lock = Path("dvc.lock")
        self.assertTrue(dvc_lock.exists(), "DVC lock file does not exist")
        self.assertGreater(dvc_lock.stat().st_size, 0, "DVC lock file is empty")


if __name__ == "__main__":
    unittest.main()
