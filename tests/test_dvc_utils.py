#!/usr/bin/env python3
"""Test the DVC utilities."""
import unittest
from pathlib import Path
import subprocess
from unittest.mock import patch, MagicMock, call

import pytest
from hypothesis import given, settings, strategies as st

from src.data.dvc_utils import (
    run_command, add_to_dvc, commit_dvc_changes, check_dvc_status
)


class TestDVCUtils(unittest.TestCase):
    """Test suite for the DVC utilities."""

    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test that run_command works correctly with a successful command."""
        # Configure mock
        mock_process = MagicMock()
        mock_process.stdout = "Command output"
        mock_run.return_value = mock_process

        # Call the function
        result = run_command("test command")

        # Verify function calls and result
        mock_run.assert_called_once_with(
            "test command",
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        self.assertEqual(result, mock_process)

    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test that run_command raises an exception when the command fails."""
        # Configure mock to raise error
        error = subprocess.CalledProcessError(1, "test command")
        error.stderr = "Command failed"
        mock_run.side_effect = error

        # Verify exception is raised
        with self.assertRaises(subprocess.CalledProcessError):
            run_command("test command")

        # Verify function calls
        mock_run.assert_called_once()

    @patch('src.data.dvc_utils.run_command')
    def test_add_to_dvc_success(self, mock_run_command):
        """Test that add_to_dvc works correctly."""
        # Configure mocks
        test_path = Path("test/path")
        
        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.return_value = True
            mock_run_command.return_value = MagicMock()

            # Call the function
            result = add_to_dvc(test_path)

            # Verify function calls and result
            mock_exists.assert_called_once()
            mock_run_command.assert_called_once_with(f"dvc add {test_path}")
            self.assertTrue(result)

    @patch('src.data.dvc_utils.run_command')
    def test_add_to_dvc_path_not_exists(self, mock_run_command):
        """Test that add_to_dvc raises a FileNotFoundError when the path doesn't exist."""
        # Configure mocks
        test_path = Path("test/path")
        
        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.return_value = False

            # Verify exception is raised
            with self.assertRaises(FileNotFoundError):
                add_to_dvc(test_path)

            # Verify function calls
            mock_exists.assert_called_once()
            mock_run_command.assert_not_called()

    @patch('src.data.dvc_utils.run_command')
    def test_add_to_dvc_command_failure(self, mock_run_command):
        """Test that add_to_dvc handles command failures correctly."""
        # Configure mocks
        test_path = Path("test/path")
        
        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.return_value = True
            error = subprocess.CalledProcessError(1, "dvc add test/path")
            mock_run_command.side_effect = error

            # Call the function
            result = add_to_dvc(test_path)

            # Verify function calls and result
            mock_exists.assert_called_once()
            mock_run_command.assert_called_once_with(f"dvc add {test_path}")
            self.assertFalse(result)

    @patch('src.data.dvc_utils.run_command')
    def test_commit_dvc_changes_success(self, mock_run_command):
        """Test that commit_dvc_changes works correctly."""
        # Configure mock
        mock_run_command.return_value = MagicMock()

        # Call the function
        result = commit_dvc_changes("Test commit")

        # Verify function calls and result
        expected_calls = [
            call("git add *.dvc data/.gitignore"),
            call('git commit -m "Test commit"')
        ]
        mock_run_command.assert_has_calls(expected_calls)
        self.assertTrue(result)

    @patch('src.data.dvc_utils.run_command')
    def test_commit_dvc_changes_failure(self, mock_run_command):
        """Test that commit_dvc_changes handles command failures correctly."""
        # Configure mock to raise error
        error = subprocess.CalledProcessError(1, "git add")
        mock_run_command.side_effect = error

        # Call the function
        result = commit_dvc_changes("Test commit")

        # Verify function calls and result
        mock_run_command.assert_called_once()
        self.assertFalse(result)

    @patch('src.data.dvc_utils.run_command')
    def test_check_dvc_status_clean(self, mock_run_command):
        """Test that check_dvc_status returns True when status is clean."""
        # Configure mock for clean status
        mock_process = MagicMock()
        mock_process.stdout = "Data and pipelines are up to date"
        mock_run_command.return_value = mock_process

        # Call the function
        result = check_dvc_status()

        # Verify function calls and result
        mock_run_command.assert_called_once_with("dvc status")
        self.assertTrue(result)

    @patch('src.data.dvc_utils.run_command')
    def test_check_dvc_status_empty(self, mock_run_command):
        """Test that check_dvc_status returns True when output is empty."""
        # Configure mock for empty output
        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run_command.return_value = mock_process

        # Call the function
        result = check_dvc_status()

        # Verify function calls and result
        mock_run_command.assert_called_once_with("dvc status")
        self.assertTrue(result)

    @patch('src.data.dvc_utils.run_command')
    def test_check_dvc_status_dirty(self, mock_run_command):
        """Test that check_dvc_status returns False when status is dirty."""
        # Configure mock for dirty status
        mock_process = MagicMock()
        mock_process.stdout = "M\tdata/raw/file.csv"
        mock_run_command.return_value = mock_process

        # Call the function
        result = check_dvc_status()

        # Verify function calls and result
        mock_run_command.assert_called_once_with("dvc status")
        self.assertFalse(result)

    @patch('src.data.dvc_utils.run_command')
    def test_check_dvc_status_error(self, mock_run_command):
        """Test that check_dvc_status handles command failures correctly."""
        # Configure mock to raise error
        error = subprocess.CalledProcessError(1, "dvc status")
        mock_run_command.side_effect = error

        # Call the function
        result = check_dvc_status()

        # Verify function calls and result
        mock_run_command.assert_called_once_with("dvc status")
        self.assertFalse(result)

    @given(
        command=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=20)
    @patch('subprocess.run')
    def test_run_command_property_based(self, mock_run, command):
        """Property-based test for run_command with various command strings."""
        # Configure mock
        mock_process = MagicMock()
        mock_process.stdout = "Command output"
        mock_run.return_value = mock_process

        # Call the function
        result = run_command(command)

        # Verify function calls
        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )

    def test_run_command_with_different_commands(self):
        """Test run_command with different command strings."""
        with patch('subprocess.run') as mock_run:
            mock_process = MagicMock()
            mock_process.stdout = "Command output"
            mock_run.return_value = mock_process
            
            # Test with a variety of command strings
            for cmd in ["echo 'hello'", "ls -la", "grep pattern file.txt"]:
                # Call the function
                result = run_command(cmd)
                
                # Verify the result
                self.assertEqual(result, mock_process)


@pytest.mark.benchmark
def test_run_command_performance(benchmark):
    """Benchmark the performance of run_command."""
    with patch('subprocess.run') as mock_run:
        mock_process = MagicMock()
        mock_process.stdout = "Command output"
        mock_run.return_value = mock_process
        
        # Benchmark the function
        result = benchmark(run_command, "echo 'hello'")
        
        # Verify the result
        assert result is mock_process


if __name__ == "__main__":
    unittest.main() 