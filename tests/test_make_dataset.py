#!/usr/bin/env python3
"""Tests for the data processing module."""
import unittest

import matplotlib
import numpy as np
import pandas as pd

from src.data.make_dataset import (
    FIGURES_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    clean_data,
    generate_eda_figures,
    load_raw_data,
    save_processed_data,
    setup_directories,
)

# Force matplotlib to use a non-GUI backend for testing
matplotlib.use('Agg')


class TestMakeDataset(unittest.TestCase):
    """Test suite for the data processing functions."""

    def setUp(self):
        """Set up the test environment."""
        # Ensure directories exist
        setup_directories()

        # Create test data if it doesn't exist
        if not list(RAW_DATA_PATH.glob("*.csv")):
            self.create_test_data()

    def create_test_data(self):
        """Create test data for testing."""
        # Create test data directory
        RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Create a test CSV file for Bitcoin
        bitcoin_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'open': np.random.uniform(40000, 45000, 10),
            'high': np.random.uniform(45000, 50000, 10),
            'low': np.random.uniform(35000, 40000, 10),
            'close': np.random.uniform(40000, 45000, 10),
            'volume': np.random.uniform(1000000, 2000000, 10),
            'market_cap': np.random.uniform(8e11, 9e11, 10)
        })
        bitcoin_data.to_csv(RAW_DATA_PATH / "bitcoin.csv", index=False)

        # Create a test CSV file for Ethereum
        eth_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'open': np.random.uniform(1500, 1600, 10),
            'high': np.random.uniform(1600, 1700, 10),
            'low': np.random.uniform(1400, 1500, 10),
            'close': np.random.uniform(1500, 1600, 10),
            'volume': np.random.uniform(500000, 600000, 10),
            'market_cap': np.random.uniform(1.8e11, 2e11, 10)
        })
        eth_data.to_csv(RAW_DATA_PATH / "ethereum.csv", index=False)

    def test_load_raw_data(self):
        """Test that raw data can be loaded correctly."""
        df = load_raw_data()

        # Check that the DataFrame is not empty
        self.assertFalse(df.empty, "DataFrame should not be empty")

        # Check that the DataFrame has the expected columns
        expected_columns = [
            'date', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'symbol'
        ]
        for column in expected_columns:
            self.assertIn(column, df.columns, f"Column {column} not found in DataFrame")

        # Check that we have at least one cryptocurrency
        self.assertGreaterEqual(
            len(df['symbol'].unique()),
            1,
            "DataFrame should have at least one cryptocurrency"
        )

    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Load raw data first
        raw_df = load_raw_data()

        # Clean the data
        cleaned_df = clean_data(raw_df)

        # Check that the cleaned DataFrame is not empty
        self.assertFalse(cleaned_df.empty, "Cleaned DataFrame should not be empty")

        # Check that date column is datetime type
        self.assertEqual(
            cleaned_df['date'].dtype,
            'datetime64[ns]',
            "Date column should be datetime type"
        )

        # Check that no rows have zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in cleaned_df.columns:
                self.assertTrue(
                    (cleaned_df[col] > 0).all(),
                    f"All values in {col} should be positive"
                )

    def test_generate_eda_figures(self):
        """Test that EDA figures are generated."""
        # Load and clean data
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)

        # Generate figures
        pdf_path = generate_eda_figures(cleaned_df)

        # Check that the PDF file exists
        self.assertTrue(pdf_path.exists(), "EDA PDF file should exist")
        self.assertGreater(pdf_path.stat().st_size, 0, "EDA PDF file should not be empty")

        # Check that individual image files exist for each cryptocurrency
        for symbol in cleaned_df['symbol'].unique():
            for fig_type in ['price_history', 'volume', 'price_distribution']:
                img_path = FIGURES_PATH / f"{symbol}_{fig_type}.png"
                self.assertTrue(img_path.exists(), f"{img_path} should exist")
                self.assertGreater(
                    img_path.stat().st_size, 0,
                    f"{img_path} should not be empty"
                )

    def test_save_processed_data(self):
        """Test that processed data is saved correctly."""
        # Load and clean data
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)

        # Save data
        parquet_path = save_processed_data(cleaned_df)

        # Check that parquet file exists
        self.assertTrue(parquet_path.exists(), "Parquet file should exist")
        self.assertGreater(
            parquet_path.stat().st_size, 0,
            "Parquet file should not be empty"
        )

        # Load the parquet file back and check schema
        df = pd.read_parquet(parquet_path)

        # Check that the DataFrame has the same shape
        self.assertEqual(df.shape, cleaned_df.shape, "Shapes should match")

        # Check that all columns are present
        for col in cleaned_df.columns:
            self.assertIn(col, df.columns, f"Column {col} not found in loaded DataFrame")

    def test_parquet_schema_and_nulls(self):
        """Test that the parquet file has the correct schema and no nulls."""
        # Run the entire pipeline first to ensure the parquet file exists
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        save_processed_data(cleaned_df)

        # Load the parquet file
        df = pd.read_parquet(PROCESSED_DATA_PATH)

        # Check schema
        expected_dtypes = {
            'date': 'datetime',
            'symbol': 'string',
            'open': 'float',
            'high': 'float',
            'low': 'float',
            'close': 'float',
            'volume': 'float',
            'market_cap': 'float'
        }

        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                # Check that column type is compatible with expected type
                if expected_dtype == 'datetime':
                    self.assertTrue(
                        pd.api.types.is_datetime64_any_dtype(df[col]),
                        f"Column {col} should be {expected_dtype} type"
                    )
                elif expected_dtype == 'string':
                    self.assertTrue(
                        pd.api.types.is_string_dtype(
                            df[col]) or pd.api.types.is_object_dtype(
                            df[col]),
                        f"Column {col} should be {expected_dtype} type")
                elif expected_dtype == 'float':
                    self.assertTrue(
                        pd.api.types.is_float_dtype(
                            df[col]) or pd.api.types.is_integer_dtype(
                            df[col]),
                        f"Column {col} should be {expected_dtype} type")

        # Check for nulls
        null_counts = df.isnull().sum()
        self.assertEqual(
            null_counts.sum(),
            0,
            f"There should be no null values in the DataFrame. Found: {null_counts}"
        )


if __name__ == "__main__":
    unittest.main()
