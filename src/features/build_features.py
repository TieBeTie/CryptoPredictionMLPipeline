#!/usr/bin/env python3
"""
Feature engineering for cryptocurrency data.

This script creates technical indicators and prediction targets for cryptocurrency price data.
"""
import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROCESSED_DATA_PATH = Path("data/processed/crypto.parquet")
FEATURES_DATA_PATH = Path("data/processed/features.parquet")
FIGURES_PATH = Path("reports/figures")


def setup_directories():
    """Create necessary directories if they don't exist."""
    FEATURES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """Load processed cryptocurrency data."""
    logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}")

    if not PROCESSED_DATA_PATH.exists():
        logger.error(f"Processed data file not found at {PROCESSED_DATA_PATH}")
        raise FileNotFoundError(f"Processed data file not found at {PROCESSED_DATA_PATH}")

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    logger.info(f"Loaded data with shape: {df.shape}")

    return df


def calculate_rolling_windows(df, windows=[7, 14, 30]):
    """Calculate rolling window features."""
    logger.info("Calculating rolling window features")

    features_df = df.copy()

    # Group by symbol to calculate rolling windows for each cryptocurrency separately
    grouped = features_df.groupby('symbol')

    for window in windows:
        # Calculate rolling mean of the close price
        features_df[f'close_rolling_mean_{window}d'] = grouped['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        # Calculate rolling standard deviation of the close price
        features_df[f'close_rolling_std_{window}d'] = grouped['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )

        # Calculate rolling mean of the volume
        features_df[f'volume_rolling_mean_{window}d'] = grouped['volume'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    logger.info(f"Added {len(windows) * 3} rolling window features")
    return features_df


def calculate_technical_indicators(df):
    """Calculate technical indicators for cryptocurrency data."""
    logger.info("Calculating technical indicators")

    features_df = df.copy()

    # Group by symbol to calculate indicators for each cryptocurrency separately
    grouped = features_df.groupby('symbol')

    # 1. Simple Moving Averages (SMA)
    for window in [5, 10, 20, 50]:
        features_df[f'sma_{window}'] = grouped['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # 2. Exponential Moving Averages (EMA)
    for window in [5, 10, 20, 50]:
        features_df[f'ema_{window}'] = grouped['close'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )

    # 3. Relative Strength Index (RSI)
    # First calculate price changes
    features_df['price_change'] = grouped['close'].transform(lambda x: x.diff())

    # Calculate gains and losses
    features_df['gain'] = features_df['price_change'].apply(lambda x: x if x > 0 else 0)
    features_df['loss'] = features_df['price_change'].apply(lambda x: -x if x < 0 else 0)

    # Calculate RSI for different windows
    for window in [6, 14]:
        # Calculate average gains and losses
        avg_gain = grouped['gain'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        avg_loss = grouped['loss'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        features_df[f'rsi_{window}'] = 100 - (100 / (1 + rs))

    # 4. Moving Average Convergence Divergence (MACD)
    features_df['macd'] = grouped['close'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean()
    )
    features_df['macd_signal'] = grouped['macd'].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']

    # 5. Bollinger Bands
    for window in [20]:
        mid_band = grouped['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        std_band = grouped['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        features_df[f'bollinger_upper_{window}'] = mid_band + (std_band * 2)
        features_df[f'bollinger_lower_{window}'] = mid_band - (std_band * 2)
        features_df[f'bollinger_pct_{window}'] = (
            features_df['close'] - features_df[f'bollinger_lower_{window}']) / (
            features_df[f'bollinger_upper_{window}'] - features_df[f'bollinger_lower_{window}'])

    # 6. Rate of Change
    for window in [5, 10, 20]:
        features_df[f'roc_{window}'] = grouped['close'].transform(
            lambda x: x.pct_change(periods=window)
        )

    # 7. Stochastic Oscillator
    for window in [14]:
        low_min = grouped['low'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        high_max = grouped['high'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        features_df[f'stoch_{window}'] = 100 * \
            (features_df['close'] - low_min) / (high_max - low_min)
        features_df[f'stoch_signal_{window}'] = features_df[f'stoch_{window}'].rolling(
            window=3, min_periods=1).mean()

    # 8. On-Balance Volume (OBV)
    features_df['price_direction'] = grouped['close'].transform(
        lambda x: np.sign(x.diff())
    )
    features_df['obv_daily'] = features_df['volume'] * features_df['price_direction']
    features_df['obv'] = grouped['obv_daily'].transform(lambda x: x.cumsum())

    # Clean up intermediate columns
    features_df = features_df.drop(
        ['price_change', 'gain', 'loss', 'price_direction', 'obv_daily'], axis=1)

    logger.info("Technical indicators calculated successfully")
    return features_df


def create_target_variables(df):
    """Create target variables for prediction."""
    logger.info("Creating target variables")

    features_df = df.copy()

    # Group by symbol to create targets for each cryptocurrency separately
    grouped = features_df.groupby('symbol')

    # 1. Next-day close price
    features_df['target_next_day_close'] = grouped['close'].transform(lambda x: x.shift(-1))

    # 2. Returns (percentage change)
    features_df['target_next_day_return'] = grouped['close'].transform(lambda x: x.pct_change(-1))

    # 3. Binary classification label (up/down)
    features_df['target_next_day_direction'] = np.sign(features_df['target_next_day_return'])
    # Convert to binary 0/1 classification (1 for up, 0 for down or no change)
    features_df['target_next_day_direction'] = features_df['target_next_day_direction'].apply(
        lambda x: 1 if x > 0 else 0
    )

    logger.info("Target variables created successfully")
    return features_df


def calculate_feature_importance(df):
    """Calculate and log feature importance baseline using Random Forest."""
    logger.info("Calculating feature importance baseline")

    # Remove rows with NaN values
    clean_df = df.dropna().copy()

    if clean_df.empty:
        logger.warning("No data available for feature importance calculation after removing NaNs")
        return None

    # Define features and target
    features = clean_df.columns.difference([
        'date', 'symbol', 'target_next_day_close', 'target_next_day_return',
        'target_next_day_direction', 'open', 'high', 'low', 'close', 'volume', 'market_cap'
    ])

    X = clean_df[features]
    y = clean_df['target_next_day_return']

    # Train a simple Random Forest model to get feature importances
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Log top features
    logger.info("Top 10 most important features:")
    for i, (feature, importance) in enumerate(
        zip(feature_importances['feature'].head(10), feature_importances['importance'].head(10))
    ):
        logger.info(f"{i+1}. {feature}: {importance:.4f}")

    # Create and save feature importance plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importances.head(20)
    )
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()

    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = FIGURES_PATH / f"feature_importance_{timestamp}.png"
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"Feature importance plot saved to {fig_path}")

    return feature_importances


def save_features_data(df):
    """Save the feature engineered data as a parquet file."""
    logger.info(f"Saving feature data to {FEATURES_DATA_PATH}")
    df.to_parquet(FEATURES_DATA_PATH, index=False)
    logger.info(f"Features saved successfully. Shape: {df.shape}")
    return FEATURES_DATA_PATH


def main():
    """Main function to run the feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline")

    # Create necessary directories
    setup_directories()

    try:
        # Load processed data
        data_df = load_processed_data()

        # Calculate rolling windows
        features_df = calculate_rolling_windows(data_df)

        # Calculate technical indicators
        features_df = calculate_technical_indicators(features_df)

        # Create target variables
        features_df = create_target_variables(features_df)

        # Log feature importance
        feature_importances = calculate_feature_importance(features_df)

        # Save features data
        save_features_data(features_df)

        logger.info("Feature engineering completed successfully")

        return features_df, feature_importances

    except Exception as e:
        logger.exception(f"Error in feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
