#!/usr/bin/env python3
"""Test DVC functionality."""
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

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
    def test_check_dvc_status_explicitly_up_to_date(self, mock_run_command):
        """Test checking DVC status when explicitly up to date."""
        mock_result = MagicMock()
        mock_result.stdout = "Data and pipelines are up to date"
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
    def test_check_dvc_status_exception(self, mock_run_command):
        """Test checking DVC status when command fails."""
        mock_run_command.side_effect = subprocess.CalledProcessError(
            returncode=1, 
            cmd="dvc status",
            stderr="Command failed"
        )

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

    @patch("src.data.dvc_utils.run_command")
    def test_commit_dvc_changes_success(self, mock_run_command):
        """Test committing DVC changes successfully."""
        result = dvc_utils.commit_dvc_changes("Test commit message")

        # Verify that run_command was called with correct arguments
        expected_calls = [
            call("git add *.dvc data/.gitignore"),
            call('git commit -m "Test commit message"')
        ]
        mock_run_command.assert_has_calls(expected_calls)
        self.assertTrue(result)

    @patch("src.data.dvc_utils.run_command")
    def test_commit_dvc_changes_git_add_fails(self, mock_run_command):
        """Test commit DVC changes when git add fails."""
        # Make the first run_command (git add) fail
        mock_run_command.side_effect = [
            subprocess.CalledProcessError(
                returncode=1, 
                cmd="git add *.dvc data/.gitignore",
                stderr="Command failed"
            )
        ]

        result = dvc_utils.commit_dvc_changes("Test commit message")

        # Verify that run_command was called only for git add
        mock_run_command.assert_called_once_with("git add *.dvc data/.gitignore")
        self.assertFalse(result)

    @patch("src.data.dvc_utils.run_command")
    def test_commit_dvc_changes_git_commit_fails(self, mock_run_command):
        """Test commit DVC changes when git commit fails."""
        # Make the second run_command (git commit) fail
        mock_run_command.side_effect = [
            MagicMock(),  # git add succeeds
            subprocess.CalledProcessError(
                returncode=1, 
                cmd='git commit -m "Test commit message"',
                stderr="Command failed"
            )
        ]

        result = dvc_utils.commit_dvc_changes("Test commit message")

        # Verify that both run_commands were called
        expected_calls = [
            call("git add *.dvc data/.gitignore"),
            call('git commit -m "Test commit message"')
        ]
        mock_run_command.assert_has_calls(expected_calls)
        self.assertFalse(result)

    @patch("src.data.dvc_utils.subprocess.run")
    @patch("src.data.dvc_utils.logger")
    def test_run_command_success(self, mock_logger, mock_subprocess_run):
        """Test successful command execution."""
        # Configure mock
        mock_result = MagicMock()
        mock_result.stdout = "Command output"
        mock_subprocess_run.return_value = mock_result

        result = dvc_utils.run_command("test command")

        # Verify subprocess.run called correctly
        mock_subprocess_run.assert_called_once_with(
            "test command",
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        self.assertEqual(result, mock_result)

    @patch("src.data.dvc_utils.subprocess.run")
    @patch("src.data.dvc_utils.logger")
    def test_run_command_failure(self, mock_logger, mock_subprocess_run):
        """Test failed command execution."""
        # Configure mock to raise exception
        error = subprocess.CalledProcessError(
            returncode=1, 
            cmd="test command",
            stderr="Command failed"
        )
        mock_subprocess_run.side_effect = error

        with self.assertRaises(subprocess.CalledProcessError):
            dvc_utils.run_command("test command")

        # Verify error logged - check for the correct error message format
        mock_logger.error.assert_any_call(f"Command failed: {error}")
        mock_logger.error.assert_any_call(f"Error output: {error.stderr}")


if __name__ == "__main__":
    unittest.main()
