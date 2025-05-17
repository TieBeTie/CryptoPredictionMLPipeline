#!/usr/bin/env python3
"""
Train and evaluate models for cryptocurrency price prediction.

This script trains an ML model, tracks experiments with MLflow, and
exports the best model to ONNX format.
"""
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from dvclive import Live
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
FEATURES_DATA_PATH = Path("data/processed/features.parquet")
MODELS_DIR = Path("models")
ONNX_MODEL_PATH = MODELS_DIR / "model.onnx"
METRICS_PATH = Path("reports/metrics.json")
FIGURES_PATH = Path("reports/figures")

# MLflow settings
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT_NAME = "crypto_price_prediction"


def setup_directories():
    """Create necessary directories if they don't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def load_feature_data():
    """Load feature engineered data."""
    logger.info(f"Loading feature data from {FEATURES_DATA_PATH}")

    if not FEATURES_DATA_PATH.exists():
        logger.error(f"Feature data file not found at {FEATURES_DATA_PATH}")
        raise FileNotFoundError(f"Feature data file not found at {FEATURES_DATA_PATH}")

    df = pd.read_parquet(FEATURES_DATA_PATH)
    logger.info(f"Loaded data with shape: {df.shape}")

    return df


def prepare_data(df, target_type="regression", test_size=0.2, valid_size=0.2):
    """
    Prepare data for model training with temporal train/validation/test splits.

    Args:
        df: DataFrame with features and target variables
        target_type: Type of prediction problem ("regression" or "classification")
        test_size: Proportion of data to use for test set
        valid_size: Proportion of training data to use for validation

    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names
    """
    logger.info("Preparing data for model training")

    # Drop rows with NaN values
    df = df.dropna()

    if df.empty:
        logger.error("No data available after removing NaN values")
        raise ValueError("No data available after removing NaN values")

    # Choose target variable based on problem type
    if target_type == "regression":
        target_col = "target_next_day_return"  # Percentage return
        logger.info(f"Using regression target: {target_col}")
    elif target_type == "classification":
        target_col = "target_next_day_direction"  # Binary up/down
        logger.info(f"Using classification target: {target_col}")
    else:
        logger.error(f"Unknown target type: {target_type}")
        raise ValueError(f"Unknown target type: {target_type}")

    # Sort by date to maintain temporal integrity
    df = df.sort_values('date')

    # Define features - exclude non-feature columns
    non_feature_cols = [
        'date', 'symbol',
        'target_next_day_close', 'target_next_day_return', 'target_next_day_direction'
    ]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Calculate split indices
    n_samples = len(df)
    n_test = int(n_samples * test_size)
    n_train_valid = n_samples - n_test
    n_valid = int(n_train_valid * valid_size)
    n_train = n_train_valid - n_valid

    # Create temporal splits
    train_df = df.iloc[:n_train]
    valid_df = df.iloc[n_train:n_train + n_valid]
    test_df = df.iloc[n_train + n_valid:]

    # Extract features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Log split sizes
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_valid.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_valid, X_test, y_train, y_valid, y_test, feature_cols


def create_model_pipeline(model_type="xgboost", params=None, target_type="regression"):
    """Create a scikit-learn pipeline with preprocessing and model."""
    logger.info(f"Creating {model_type} model pipeline for {target_type}")

    if params is None:
        params = {}

    # Define preprocessing steps
    preprocessing_steps = [
        ('scaler', StandardScaler())
    ]

    # Define model
    if model_type == "xgboost":
        if target_type == "regression":
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **params
            )
        else:  # classification
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **params
            )
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

    # Create pipeline
    pipeline = Pipeline([
        *preprocessing_steps,
        ('model', model)
    ])

    return pipeline


def train_and_evaluate_model(
    X_train, X_valid, X_test, y_train, y_valid, y_test,
    model_pipeline, target_type="regression"
):
    """Train and evaluate model, log metrics with MLflow and DVCLive."""
    logger.info("Training and evaluating model")

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Set up DVCLive for metrics tracking
    live = Live("reports")

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Train model
        logger.info("Training model")
        model_pipeline.fit(X_train, y_train)

        # Get predictions on all sets
        train_preds = model_pipeline.predict(X_train)
        valid_preds = model_pipeline.predict(X_valid)
        test_preds = model_pipeline.predict(X_test)

        if target_type == "regression":
            # Calculate regression metrics
            metrics = calculate_regression_metrics(
                y_train, train_preds,
                y_valid, valid_preds,
                y_test, test_preds
            )
        else:  # classification
            # Calculate classification metrics
            metrics = calculate_classification_metrics(
                y_train, train_preds,
                y_valid, valid_preds,
                y_test, test_preds
            )

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log metrics to DVCLive
        for metric_name, metric_value in metrics.items():
            live.log_metric(metric_name, metric_value)

        # Plot feature importance if possible
        if hasattr(model_pipeline.named_steps['model'], 'feature_importances_'):
            feature_importances = plot_feature_importance(
                model_pipeline.named_steps['model'].feature_importances_,
                feature_names=X_train.columns
            )
            mlflow.log_artifact(feature_importances)
            live.log_image(str(feature_importances), "feature_importances.png")

        # If it's classification, plot confusion matrix
        if target_type == "classification":
            cm_fig_path = plot_confusion_matrix(y_test, test_preds)
            mlflow.log_artifact(cm_fig_path)
            live.log_image(str(cm_fig_path), "confusion_matrix.png")

        # Log the model
        mlflow.sklearn.log_model(model_pipeline, "model")

        # Save metrics to JSON
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Finalize DVCLive
        live.end()

        return model_pipeline, run_id, metrics


def calculate_regression_metrics(y_train, train_preds, y_valid, valid_preds, y_test, test_preds):
    """Calculate regression metrics for all data splits."""
    metrics = {}

    # Training metrics
    metrics['train_rmse'] = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    metrics['train_mae'] = float(mean_absolute_error(y_train, train_preds))
    metrics['train_r2'] = float(r2_score(y_train, train_preds))

    # Validation metrics
    metrics['valid_rmse'] = float(np.sqrt(mean_squared_error(y_valid, valid_preds)))
    metrics['valid_mae'] = float(mean_absolute_error(y_valid, valid_preds))
    metrics['valid_r2'] = float(r2_score(y_valid, valid_preds))

    # Test metrics
    metrics['test_rmse'] = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    metrics['test_mae'] = float(mean_absolute_error(y_test, test_preds))
    metrics['test_r2'] = float(r2_score(y_test, test_preds))

    # Log all metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    return metrics


def calculate_classification_metrics(
        y_train,
        train_preds,
        y_valid,
        valid_preds,
        y_test,
        test_preds):
    """Calculate classification metrics for all data splits."""
    metrics = {}

    # Training metrics
    metrics['train_accuracy'] = float(np.mean(y_train == train_preds))
    metrics['train_precision'] = float(precision_score(y_train, train_preds, zero_division=0))
    metrics['train_recall'] = float(recall_score(y_train, train_preds, zero_division=0))
    metrics['train_f1'] = float(f1_score(y_train, train_preds, zero_division=0))

    # Validation metrics
    metrics['valid_accuracy'] = float(np.mean(y_valid == valid_preds))
    metrics['valid_precision'] = float(precision_score(y_valid, valid_preds, zero_division=0))
    metrics['valid_recall'] = float(recall_score(y_valid, valid_preds, zero_division=0))
    metrics['valid_f1'] = float(f1_score(y_valid, valid_preds, zero_division=0))

    # Test metrics
    metrics['test_accuracy'] = float(np.mean(y_test == test_preds))
    metrics['test_precision'] = float(precision_score(y_test, test_preds, zero_division=0))
    metrics['test_recall'] = float(recall_score(y_test, test_preds, zero_division=0))
    metrics['test_f1'] = float(f1_score(y_test, test_preds, zero_division=0))

    # Log detailed classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, test_preds))

    # Log all metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    return metrics


def plot_feature_importance(importances, feature_names):
    """Plot feature importance and save to file."""
    # Create DataFrame for importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='importance',
        y='feature',
        data=importance_df.head(20)
    )
    plt.title('Feature Importance')
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = FIGURES_PATH / f"model_feature_importance_{timestamp}.png"
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix and save to file."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = FIGURES_PATH / f"confusion_matrix_{timestamp}.png"
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def get_git_commit_hash():
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Could not get git commit hash, using placeholder")
        return "unknown-commit-hash"


def export_model_to_onnx(model, feature_count, run_id):
    """Export scikit-learn pipeline to ONNX format with metadata."""
    logger.info("Exporting model to ONNX format")

    try:
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, feature_count]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12,
            options={id(model): {'zipmap': False}}
        )

        # Add metadata
        commit_hash = get_git_commit_hash()
        saved_date = datetime.now(timezone.utc).isoformat()

        metadata = {
            "git_commit_hash": commit_hash,
            "save_date": saved_date,
            "mlflow_run_id": run_id,
            "description": "Cryptocurrency price prediction model"
        }

        # Add metadata to ONNX model
        meta = onnx_model.metadata_props
        for key, value in metadata.items():
            meta.append(
                lambda: None  # lambda function that returns None, placeholder
            ).key = key
            meta[-1].value = str(value)  # set value of the last element

        # Save ONNX model
        with open(ONNX_MODEL_PATH, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Model exported to {ONNX_MODEL_PATH}")

        return ONNX_MODEL_PATH

    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")
        raise


def main(target_type="regression", model_type="xgboost", params=None):
    """Main function to train and evaluate models."""
    logger.info("Starting model training pipeline")

    # Create necessary directories
    setup_directories()

    try:
        # Load feature data
        df = load_feature_data()

        # Prepare data with temporal splits
        X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names = prepare_data(
            df, target_type=target_type
        )

        # Create model pipeline
        model_pipeline = create_model_pipeline(
            model_type=model_type, params=params, target_type=target_type
        )

        # Train and evaluate model
        trained_model, run_id, metrics = train_and_evaluate_model(
            X_train, X_valid, X_test,
            y_train, y_valid, y_test,
            model_pipeline, target_type=target_type
        )

        # Export model to ONNX
        onnx_path = export_model_to_onnx(
            trained_model,
            feature_count=len(feature_names),
            run_id=run_id
        )

        logger.info("Model training pipeline completed successfully")

        return trained_model, metrics, onnx_path

    except Exception as e:
        logger.exception(f"Error in model training pipeline: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train cryptocurrency price prediction model')
    parser.add_argument('--target-type', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Type of prediction problem (regression or classification)')
    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost'],
                        help='Type of model to use')
    parser.add_argument('--param', action='append', nargs=2, metavar=('KEY', 'VALUE'),
                        help='Model hyperparameter in the format key value')

    args = parser.parse_args()

    # Parse model parameters if provided
    params = {}
    if args.param:
        for key, value in args.param:
            # Try to convert value to numeric if possible
            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

    # Run main function with parsed arguments
    main(target_type=args.target_type, model_type=args.model_type, params=params)
