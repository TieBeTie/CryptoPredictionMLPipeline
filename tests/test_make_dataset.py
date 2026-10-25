#!/usr/bin/env python3
"""Tests for the data processing module."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.data.make_dataset import (
    FIGURES_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    clean_data,
    generate_eda_figures,
    load_raw_data,
    main,
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

    # New test cases for improved coverage:

    def test_load_raw_data_no_files(self):
        """Test error handling when no CSV files are found."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Patch the RAW_DATA_PATH with the temporary directory
            with patch('src.data.make_dataset.RAW_DATA_PATH', Path(tmpdirname)):
                # No CSV files exist in the temporary directory
                with self.assertRaises(FileNotFoundError):
                    load_raw_data()

    @patch('pandas.read_csv')
    def test_load_raw_data_with_error(self, mock_read_csv):
        """Test error handling when reading a CSV file fails."""
        # Set up the mock to raise an exception
        mock_read_csv.side_effect = Exception("Error reading CSV")

        # Create a temporary directory with a test CSV file
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a dummy CSV file
            temp_path = Path(tmpdirname)
            (temp_path / "test.csv").touch()
            
            # Patch the RAW_DATA_PATH
            with patch('src.data.make_dataset.RAW_DATA_PATH', temp_path):
                # This should now raise a ValueError because no DataFrames could be created
                with self.assertRaises(ValueError):
                    load_raw_data()

    def test_clean_data_with_missing_values(self):
        """Test data cleaning when there are missing values."""
        # Create a DataFrame with missing values
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [1.0, 2.0, np.nan, 4.0, 5.0],
            'high': [1.5, 2.5, 3.5, np.nan, 5.5],
            'low': [0.5, 1.5, np.nan, 3.5, 4.5],
            'close': [1.2, 2.2, 3.2, 4.2, np.nan],
            'volume': [1000, np.nan, 3000, 4000, 5000],
            'market_cap': [1e6, 2e6, 3e6, 4e6, np.nan],
            'symbol': ['BTC', 'BTC', 'BTC', 'BTC', 'BTC']
        })

        # Clean the data
        cleaned_df = clean_data(df)

        # Check that the cleaned DataFrame has no missing values
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0, 
                        "Cleaned DataFrame should have no missing values")

    def test_clean_data_with_non_numeric_data(self):
        """Test data cleaning when there are non-numeric values that need coercion."""
        # Create a DataFrame with non-numeric values in numeric columns
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [1.0, '2.0', 3.0, '4.0', 5.0],  # Mix of strings and numbers
            'high': [1.5, 2.5, '3.5', 4.5, '5.5'],
            'low': ['0.5', 1.5, 2.5, '3.5', 4.5],
            'close': [1.2, '2.2', 3.2, '4.2', 5.2],
            'volume': ['1000', 2000, '3000', 4000, '5000'],
            'market_cap': [1e6, '2e6', 3e6, '4e6', 5e6],
            'symbol': ['BTC', 'BTC', 'BTC', 'BTC', 'BTC']
        })

        # Clean the data
        cleaned_df = clean_data(df)

        # Check that all numeric columns are actually numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df[col]),
                          f"Column {col} should be numeric after cleaning")

    def test_clean_data_with_zero_prices(self):
        """Test that rows with zero prices are removed."""
        # Create a DataFrame with zero prices
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [1.0, 0.0, 3.0, 4.0, 5.0],  # Zero price
            'high': [1.5, 2.5, 0.0, 4.5, 5.5],  # Zero price
            'low': [0.5, 1.5, 2.5, 0.0, 4.5],   # Zero price
            'close': [1.2, 2.2, 3.2, 4.2, 0.0], # Zero price
            'volume': [1000, 2000, 3000, 4000, 5000],
            'market_cap': [1e6, 2e6, 3e6, 4e6, 5e6],
            'symbol': ['BTC', 'BTC', 'BTC', 'BTC', 'BTC']
        })

        # Clean the data
        cleaned_df = clean_data(df)

        # Check that rows with zero prices are removed
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            self.assertTrue((cleaned_df[col] > 0).all(), 
                          f"All values in {col} should be positive after cleaning")
        
        # Check that the DataFrame has fewer rows than the original
        self.assertLess(len(cleaned_df), len(df), 
                       "Cleaned DataFrame should have fewer rows than original")

    def test_clean_data_with_negative_prices(self):
        """Test that rows with negative prices are removed."""
        # Create a DataFrame with negative prices
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [1.0, -2.0, 3.0, 4.0, 5.0],  # Negative price
            'high': [1.5, 2.5, -3.5, 4.5, 5.5],  # Negative price
            'low': [0.5, 1.5, 2.5, -3.5, 4.5],   # Negative price
            'close': [1.2, 2.2, 3.2, 4.2, -5.2], # Negative price
            'volume': [1000, 2000, 3000, 4000, 5000],
            'market_cap': [1e6, 2e6, 3e6, 4e6, 5e6],
            'symbol': ['BTC', 'BTC', 'BTC', 'BTC', 'BTC']
        })

        # Clean the data
        cleaned_df = clean_data(df)

        # Check that rows with negative prices are removed
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            self.assertTrue((cleaned_df[col] > 0).all(), 
                          f"All values in {col} should be positive after cleaning")
        
        # Check that the DataFrame has fewer rows than the original
        self.assertLess(len(cleaned_df), len(df), 
                       "Cleaned DataFrame should have fewer rows than original")

    def test_main_function(self):
        """Test the main function that runs the entire pipeline."""
        # Run the main function
        result = main()
        
        # Check the result
        self.assertIsInstance(result, tuple, "Result should be a tuple")
        self.assertEqual(len(result), 3, "Result should have 3 elements")
        
        cleaned_df, pdf_path, parquet_path = result
        
        # Check that the DataFrame is not empty
        self.assertFalse(cleaned_df.empty, "DataFrame should not be empty")
        
        # Check that the PDF file exists
        self.assertTrue(pdf_path.exists(), "PDF file should exist")
        self.assertGreater(pdf_path.stat().st_size, 0, "PDF file should not be empty")
        
        # Check that the parquet file exists
        self.assertTrue(parquet_path.exists(), "Parquet file should exist")
        self.assertGreater(parquet_path.stat().st_size, 0, "Parquet file should not be empty")

    @patch('src.data.make_dataset.load_raw_data')
    def test_main_function_with_exception(self, mock_load_raw_data):
        """Test error handling in the main function."""
        # Configure the mock to raise an exception
        mock_load_raw_data.side_effect = Exception("Test error")
        
        # This should raise the exception
        with self.assertRaises(Exception):
            main()

    @given(
        missing_pct=st.floats(min_value=0.0, max_value=0.5),
        n_rows=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=10)
    def test_property_based_missing_values(self, missing_pct, n_rows):
        """Property-based test for handling missing values with different percentages."""
        # Create test data with missing values
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=n_rows),
            'open': np.random.uniform(100, 1000, n_rows),
            'high': np.random.uniform(100, 1000, n_rows),
            'low': np.random.uniform(100, 1000, n_rows),
            'close': np.random.uniform(100, 1000, n_rows),
            'volume': np.random.uniform(1000, 10000, n_rows),
            'market_cap': np.random.uniform(1e6, 1e7, n_rows),
            'symbol': ['TEST'] * n_rows
        })
        
        # Add missing values according to the percentage
        mask = np.random.random(df.shape) < missing_pct
        df = df.mask(mask)
        
        # Clean the data
        cleaned_df = clean_data(df)
        
        # Check that the cleaned DataFrame has no missing values
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0, 
                        "Cleaned DataFrame should have no missing values")


@pytest.mark.benchmark
def test_setup_directories_performance(benchmark):
    """Test the performance of the setup_directories function."""
    # Use benchmark to measure the function's performance
    benchmark(setup_directories)


@pytest.mark.benchmark
def test_clean_data_performance(benchmark):
    """Test the performance of the clean_data function."""
    # Create a larger test DataFrame
    n_rows = 1000
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=n_rows),
        'open': np.random.uniform(100, 1000, n_rows),
        'high': np.random.uniform(100, 1000, n_rows),
        'low': np.random.uniform(100, 1000, n_rows),
        'close': np.random.uniform(100, 1000, n_rows),
        'volume': np.random.uniform(1000, 10000, n_rows),
        'market_cap': np.random.uniform(1e6, 1e7, n_rows),
        'symbol': np.random.choice(['BTC', 'ETH', 'XRP'], n_rows)
    })
    
    # Add some missing values
    mask = np.random.random(df.shape) < 0.05  # 5% missing
    df = df.mask(mask)
    
    # Use benchmark to measure the function's performance
    benchmark(lambda: clean_data(df))


@pytest.mark.benchmark
def test_save_processed_data_performance(benchmark):
    """Test the performance of the save_processed_data function."""
    # Create a test DataFrame
    n_rows = 1000
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=n_rows),
        'open': np.random.uniform(100, 1000, n_rows),
        'high': np.random.uniform(100, 1000, n_rows),
        'low': np.random.uniform(100, 1000, n_rows),
        'close': np.random.uniform(100, 1000, n_rows),
        'volume': np.random.uniform(1000, 10000, n_rows),
        'market_cap': np.random.uniform(1e6, 1e7, n_rows),
        'symbol': np.random.choice(['BTC', 'ETH', 'XRP'], n_rows)
    })
    
    # Use benchmark to measure the function's performance
    benchmark(lambda: save_processed_data(df))


if __name__ == "__main__":
    unittest.main()
