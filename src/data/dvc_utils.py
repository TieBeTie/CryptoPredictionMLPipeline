#!/usr/bin/env python3
"""
Utilities for DVC operations.

This module provides helper functions for working with DVC.
"""
import logging
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        logger.info(f"Command output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise


def add_to_dvc(path):
    """Add a file or directory to DVC tracking."""
    path = Path(path)
    logger.info(f"Adding {path} to DVC")

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        raise FileNotFoundError(f"Path does not exist: {path}")

    try:
        run_command(f"dvc add {path}")
        logger.info(f"Successfully added {path} to DVC")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add {path} to DVC: {e}")
        return False


def commit_dvc_changes(message):
    """Commit DVC changes to Git."""
    logger.info("Committing DVC changes to Git")

    try:
        # Add DVC files to git
        run_command("git add *.dvc data/.gitignore")

        # Commit DVC files
        run_command(f'git commit -m "{message}"')

        logger.info("Successfully committed DVC changes to Git")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to commit DVC changes: {e}")
        return False


def check_dvc_status():
    """Check if DVC status is clean."""
    logger.info("Checking DVC status")

    try:
        result = run_command("dvc status")

        # Check if status indicates everything is up to date or if the output is empty
        if "Data and pipelines are up to date" in result.stdout or result.stdout.strip() == "":
            logger.info("DVC status is clean")
            return True
        else:
            logger.warning(f"DVC status has uncommitted changes: {result.stdout}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check DVC status: {e}")
        return False
