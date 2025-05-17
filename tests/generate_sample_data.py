#!/usr/bin/env python3
"""
Generate sample data for testing the model training pipeline.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(output_path, n_samples=1000):
    """Generate sample cryptocurrency data with features."""
    # Create dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create price data
    price = 10000 + np.cumsum(np.random.normal(0, 200, n_samples))

    # Create features
    df = pd.DataFrame({
        'date': dates,
        'symbol': 'BTC',
        'close': price,
        'volume': np.random.normal(1000000, 200000, n_samples),
        'open': price * (1 + np.random.normal(0, 0.01, n_samples)),
        'high': price * (1 + np.random.normal(0.01, 0.005, n_samples)),
        'low': price * (1 - np.random.normal(0.01, 0.005, n_samples)),
    })

    # Create technical indicators as features
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi_14'] = np.random.uniform(0, 100, n_samples)  # Simplified RSI
    df['macd'] = df['sma_5'] - df['sma_20']
    df['volatility'] = df['close'].rolling(10).std()
    df['price_change_1d'] = df['close'].pct_change()
    df['volume_change_1d'] = df['volume'].pct_change()

    # Create target variables
    df['target_next_day_close'] = df['close'].shift(-1)
    df['target_next_day_return'] = df['target_next_day_close'] / df['close'] - 1
    df['target_next_day_direction'] = (df['target_next_day_return'] > 0).astype(int)

    # Drop NaN values
    df = df.dropna()

    # Create feature columns
    feature_cols = [
        'sma_5', 'sma_20', 'rsi_14', 'macd', 'volatility',
        'price_change_1d', 'volume_change_1d'
    ]

    # Add some random features
    for i in range(1, 6):
        col_name = f'feature{i}'
        df[col_name] = np.random.normal(0, 1, len(df))
        feature_cols.append(col_name)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to parquet
    df.to_parquet(output_path)

    print(f"Generated sample data with {len(df)} rows and saved to {output_path}")
    print(f"Features: {feature_cols}")

    return df


if __name__ == "__main__":
    output_path = Path("data/processed/features.parquet")
    generate_sample_data(output_path)
