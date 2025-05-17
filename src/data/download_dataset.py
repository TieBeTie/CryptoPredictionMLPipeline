#!/usr/bin/env python3
"""
Download cryptocurrency data from Kaggle.

This script downloads the "All Crypto Currencies" dataset from Kaggle
and extracts it to the data/raw directory.
"""
import logging
import os
from pathlib import Path

import kaggle
from dotenv import load_dotenv

from src.data.dvc_utils import add_to_dvc, check_dvc_status

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
DATASET_NAME = "jessevent/all-crypto-currencies"
RAW_DATA_PATH = Path("data/raw")


def create_directories():
    """Create necessary directories if they don't exist."""
    logger.info("Creating directories...")
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


def download_dataset():
    """Download dataset from Kaggle and extract it to the raw data directory."""
    # Load environment variables
    load_dotenv()

    # Check if Kaggle API credentials are set
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        logger.error("Kaggle API credentials not found in environment variables")
        logger.error("Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
        raise EnvironmentError("Kaggle API credentials not found")

    # Download the dataset
    logger.info(f"Downloading dataset: {DATASET_NAME}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset=DATASET_NAME,
        path=RAW_DATA_PATH,
        unzip=True,
    )

    # Verify the download
    files = list(RAW_DATA_PATH.glob("*.csv"))
    logger.info(f"Downloaded {len(files)} files to {RAW_DATA_PATH}")

    for file in files:
        logger.info(f"Found file: {file.name} ({file.stat().st_size / 1024:.2f} KB)")

    return files


def track_with_dvc():
    """Track raw data with DVC."""
    logger.info("Tracking data with DVC...")

    try:
        # Add the raw data directory to DVC
        add_to_dvc(RAW_DATA_PATH)

        # Check DVC status
        is_clean = check_dvc_status()
        if is_clean:
            logger.info("DVC tracking successful. Data is tracked.")
        else:
            logger.warning("DVC tracking may have issues. Check DVC status.")

        return is_clean
    except Exception as e:
        logger.exception(f"Error tracking data with DVC: {e}")
        return False


def main():
    """Main function to download and extract the dataset."""
    logger.info("Starting dataset download process")
    create_directories()

    try:
        files = download_dataset()
        logger.info(f"Successfully downloaded {len(files)} files")

        # Track the raw data with DVC
        track_with_dvc()
    except Exception as e:
        logger.exception(f"Error downloading dataset: {e}")
        return False

    logger.info("Download process completed successfully")
    return True


if __name__ == "__main__":
    main()
