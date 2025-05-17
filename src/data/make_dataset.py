#!/usr/bin/env python3
"""
Process raw cryptocurrency data.

This script takes raw cryptocurrency data, cleans it, and saves it as a parquet file.
"""
import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
RAW_DATA_PATH = Path("data/raw/test")
PROCESSED_DATA_PATH = Path("data/processed/crypto.parquet")
FIGURES_PATH = Path("reports/figures")


def setup_directories():
    """Create necessary directories if they don't exist."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    """Load raw cryptocurrency data from CSV files."""
    logger.info("Loading raw data from CSV files")

    # Get all CSV files in the raw data directory
    csv_files = list(RAW_DATA_PATH.glob("*.csv"))

    if not csv_files:
        logger.error("No CSV files found in %s", RAW_DATA_PATH)
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_PATH}")

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through each CSV file
    for file_path in csv_files:
        logger.info("Loading file: %s", file_path.name)

        # Extract cryptocurrency name from filename
        crypto_name = file_path.stem

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Add cryptocurrency name as a column
            df['symbol'] = crypto_name

            # Append to our list of DataFrames
            dataframes.append(df)
        except Exception as e:
            logger.error("Error loading file %s: %s", file_path.name, e)

    # Combine all DataFrames into one
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info("Successfully loaded %d cryptocurrencies", len(dataframes))
        logger.info("Combined data shape: %s", combined_df.shape)
        return combined_df
    else:
        logger.error("No data loaded")
        raise ValueError("No data could be loaded from CSV files")


def clean_data(df):
    """Clean and preprocess the raw data."""
    logger.info("Cleaning and preprocessing data")

    # Make a copy to avoid modifying the input
    df = df.copy()

    # Convert date column to datetime
    logger.info("Converting date column to datetime")
    df['date'] = pd.to_datetime(df['date'])

    # Sort by symbol and date
    logger.info("Sorting data by symbol and date")
    df = df.sort_values(['symbol', 'date'])

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning("Missing values found: %s", missing_values[missing_values > 0])

        # For this example, we'll fill missing values
        # In a real application, you might want a more sophisticated approach
        logger.info("Filling missing values")

        # Fill missing values with forward fill then backward fill
        # This preserves time series integrity
        df = df.groupby('symbol').apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

        # Reset index that gets created by groupby
        df = df.reset_index(drop=True)

    # Ensure all numeric columns are actually numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Additional data cleaning steps
    # Remove rows with unreasonable values (e.g., zero or negative prices)
    logger.info("Removing rows with invalid price values")
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            df = df[df[col] > 0]

    logger.info("Data cleaning completed. Shape after cleaning: %s", df.shape)
    return df


def generate_eda_figures(df):
    """Generate exploratory data analysis figures."""
    logger.info("Generating EDA figures")

    # Create a timestamp for the report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = FIGURES_PATH / f"eda_report_{timestamp}.pdf"

    with PdfPages(pdf_path) as pdf:
        # Figure 1: Price Time Series for each cryptocurrency
        for symbol in df['symbol'].unique():
            crypto_df = df[df['symbol'] == symbol]

            plt.figure(figsize=(10, 6))
            plt.plot(crypto_df['date'], crypto_df['close'], label='Close Price')
            plt.title(f'{symbol} Price History')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Save an individual image too
            img_path = FIGURES_PATH / f"{symbol}_price_history.png"
            plt.figure(figsize=(10, 6))
            plt.plot(crypto_df['date'], crypto_df['close'], label='Close Price')
            plt.title(f'{symbol} Price History')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()

        # Figure 2: Volume Analysis
        for symbol in df['symbol'].unique():
            crypto_df = df[df['symbol'] == symbol]

            plt.figure(figsize=(10, 6))
            plt.bar(crypto_df['date'], crypto_df['volume'])
            plt.title(f'{symbol} Trading Volume')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Save an individual image too
            img_path = FIGURES_PATH / f"{symbol}_volume.png"
            plt.figure(figsize=(10, 6))
            plt.bar(crypto_df['date'], crypto_df['volume'])
            plt.title(f'{symbol} Trading Volume')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()

        # Figure 3: Price distribution
        for symbol in df['symbol'].unique():
            crypto_df = df[df['symbol'] == symbol]

            plt.figure(figsize=(10, 6))
            plt.hist(crypto_df['close'], bins=30)
            plt.title(f'{symbol} Price Distribution')
            plt.xlabel('Price (USD)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Save an individual image too
            img_path = FIGURES_PATH / f"{symbol}_price_distribution.png"
            plt.figure(figsize=(10, 6))
            plt.hist(crypto_df['close'], bins=30)
            plt.title(f'{symbol} Price Distribution')
            plt.xlabel('Price (USD)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()

    logger.info(f"EDA report saved to {pdf_path}")
    return pdf_path


def save_processed_data(df):
    """Save the processed data as a parquet file."""
    logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}")
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Data saved successfully. Shape: {df.shape}")
    return PROCESSED_DATA_PATH


def main():
    """Main function to run the data processing pipeline."""
    logger.info("Starting data processing pipeline")

    # Create necessary directories
    setup_directories()

    try:
        # Load raw data
        raw_df = load_raw_data()

        # Clean the data
        cleaned_df = clean_data(raw_df)

        # Generate EDA figures
        pdf_path = generate_eda_figures(cleaned_df)

        # Save processed data
        parquet_path = save_processed_data(cleaned_df)

        logger.info("Data processing completed successfully")

        return cleaned_df, pdf_path, parquet_path

    except Exception as e:
        logger.exception(f"Error in data processing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
