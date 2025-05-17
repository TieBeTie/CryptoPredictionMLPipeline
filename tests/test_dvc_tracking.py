#!/usr/bin/env python3
"""Test DVC functionality."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.data.dvc_utils as dvc_utils


class TestDVCUtils(unittest.TestCase):
    """Test suite for DVC utilities."""

    @patch("src.data.dvc_utils.run_command")
    def test_add_to_dvc(self, mock_run_command):
        """Test adding a path to DVC."""
        # Set up test directory
        test_path = Path("test_dir")

        # Mock Path.exists to return True
        with patch.object(Path, "exists", return_value=True):
            result = dvc_utils.add_to_dvc(test_path)

            # Verify that run_command was called with correct arguments
            mock_run_command.assert_called_once_with(f"dvc add {test_path}")
            self.assertTrue(result)

    @patch("src.data.dvc_utils.run_command")
    def test_check_dvc_status_clean(self, mock_run_command):
        """Test checking DVC status when clean."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_run_command.return_value = mock_result

        result = dvc_utils.check_dvc_status()

        # Verify that run_command was called with correct arguments
        mock_run_command.assert_called_once_with("dvc status")
        self.assertTrue(result)

    @patch("src.data.dvc_utils.run_command")
    def test_check_dvc_status_not_clean(self, mock_run_command):
        """Test checking DVC status when not clean."""
        mock_result = MagicMock()
        mock_result.stdout = "data/raw.dvc: changed"
        mock_run_command.return_value = mock_result

        result = dvc_utils.check_dvc_status()

        # Verify that run_command was called with correct arguments
        mock_run_command.assert_called_once_with("dvc status")
        self.assertFalse(result)

    @patch("src.data.dvc_utils.run_command")
    def test_add_to_dvc_file_not_found(self, mock_run_command):
        """Test adding a non-existent path to DVC."""
        # Set up test directory
        test_path = Path("non_existent_path")

        # Mock Path.exists to return False
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                dvc_utils.add_to_dvc(test_path)

            # Verify that run_command was not called
            mock_run_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
