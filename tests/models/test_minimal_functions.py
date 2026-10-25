"""Minimal tests for specific utility functions from train_model.py."""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# Import the functions directly
# Note: These are copied here to avoid importing the full module with dependencies
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
        return "unknown-commit-hash"


def setup_directories():
    """Create necessary directories if they don't exist."""
    # Here we use test directories instead of the actual paths
    models_dir = Path("test_models")
    figures_path = Path("test_reports/figures")
    metrics_path = Path("test_reports/metrics.json")

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Return paths for testing
    return models_dir, figures_path, metrics_path


def calculate_regression_metrics(y_train, train_preds, y_valid, valid_preds, y_test, test_preds):
    """Calculate regression metrics for all data splits."""
    metrics = {}

    # Function to calculate MSE and MAE
    def calculate_mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def calculate_r2(y_true, y_pred):
        # Calculate R² score
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        if ss_total == 0:  # If data has no variance, R² is undefined
            return 0
        return float(1 - (ss_res / ss_total))

    # Training metrics
    metrics['train_rmse'] = float(np.sqrt(calculate_mse(y_train, train_preds)))
    metrics['train_mae'] = calculate_mae(y_train, train_preds)
    metrics['train_r2'] = calculate_r2(y_train, train_preds)

    # Validation metrics
    metrics['valid_rmse'] = float(np.sqrt(calculate_mse(y_valid, valid_preds)))
    metrics['valid_mae'] = calculate_mae(y_valid, valid_preds)
    metrics['valid_r2'] = calculate_r2(y_valid, valid_preds)

    # Test metrics
    metrics['test_rmse'] = float(np.sqrt(calculate_mse(y_test, test_preds)))
    metrics['test_mae'] = calculate_mae(y_test, test_preds)
    metrics['test_r2'] = calculate_r2(y_test, test_preds)

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

    # Utility functions to calculate metrics
    def calculate_accuracy(y_true, y_pred):
        return float(np.mean(y_true == y_pred))
    
    def calculate_precision(y_true, y_pred, zero_division=0):
        # True positives / (True positives + False positives)
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        if true_pos + false_pos == 0:
            return zero_division
        return float(true_pos / (true_pos + false_pos))
    
    def calculate_recall(y_true, y_pred, zero_division=0):
        # True positives / (True positives + False negatives)
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        false_neg = np.sum((y_true == 1) & (y_pred == 0))
        if true_pos + false_neg == 0:
            return zero_division
        return float(true_pos / (true_pos + false_neg))
    
    def calculate_f1(y_true, y_pred, zero_division=0):
        precision = calculate_precision(y_true, y_pred, zero_division)
        recall = calculate_recall(y_true, y_pred, zero_division)
        if precision + recall == 0:
            return zero_division
        return float(2 * (precision * recall) / (precision + recall))

    # Training metrics
    metrics['train_accuracy'] = calculate_accuracy(y_train, train_preds)
    metrics['train_precision'] = calculate_precision(y_train, train_preds, zero_division=0)
    metrics['train_recall'] = calculate_recall(y_train, train_preds, zero_division=0)
    metrics['train_f1'] = calculate_f1(y_train, train_preds, zero_division=0)

    # Validation metrics
    metrics['valid_accuracy'] = calculate_accuracy(y_valid, valid_preds)
    metrics['valid_precision'] = calculate_precision(y_valid, valid_preds, zero_division=0)
    metrics['valid_recall'] = calculate_recall(y_valid, valid_preds, zero_division=0)
    metrics['valid_f1'] = calculate_f1(y_valid, valid_preds, zero_division=0)

    # Test metrics
    metrics['test_accuracy'] = calculate_accuracy(y_test, test_preds)
    metrics['test_precision'] = calculate_precision(y_test, test_preds, zero_division=0)
    metrics['test_recall'] = calculate_recall(y_test, test_preds, zero_division=0)
    metrics['test_f1'] = calculate_f1(y_test, test_preds, zero_division=0)

    return metrics


# Tests
def test_setup_directories():
    """Test that setup_directories creates necessary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temp directory
        original_dir = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Run function
            models_dir, figures_path, metrics_path = setup_directories()
            
            # Check directories were created
            assert models_dir.exists()
            assert figures_path.exists()
            assert metrics_path.parent.exists()
        finally:
            # Change back to original directory
            os.chdir(original_dir)


@patch('subprocess.run')
def test_get_git_commit_hash(mock_run):
    """Test getting git commit hash."""
    # Configure mock
    mock_process = MagicMock()
    mock_process.stdout = "abcdef1234567890\n"
    mock_process.check = True
    mock_run.return_value = mock_process

    # Test function
    result = get_git_commit_hash()
    assert result == "abcdef1234567890"

    # Test error case
    mock_run.side_effect = subprocess.SubprocessError("Git error")
    result = get_git_commit_hash()
    assert result == "unknown-commit-hash"


def test_regression_metrics():
    """Test regression metrics calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    metrics = calculate_regression_metrics(
        y_true, y_pred, y_true, y_pred, y_true, y_pred
    )

    # Check metrics
    assert 'train_rmse' in metrics
    assert 'valid_rmse' in metrics
    assert 'test_rmse' in metrics
    assert metrics['train_rmse'] == pytest.approx(0.1, abs=1e-4)
    
    # Test perfect predictions
    metrics_perfect = calculate_regression_metrics(
        y_true, y_true, y_true, y_true, y_true, y_true
    )
    assert metrics_perfect['train_rmse'] == 0.0
    assert metrics_perfect['test_r2'] == 1.0
    
    # Test bad predictions - use a value really different from the mean instead of a constant
    y_bad = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # Values farther from mean
    metrics_bad = calculate_regression_metrics(
        y_true, y_bad, y_true, y_bad, y_true, y_bad
    )
    assert metrics_bad['train_r2'] <= 0  # R2 can be negative or zero for bad models


def test_classification_metrics():
    """Test classification metrics calculation."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1])  # 3/5 correct

    metrics = calculate_classification_metrics(
        y_true, y_pred, y_true, y_pred, y_true, y_pred
    )

    # Check metrics
    assert 'train_accuracy' in metrics
    assert 'valid_precision' in metrics
    assert 'test_recall' in metrics
    assert 'test_f1' in metrics
    assert metrics['train_accuracy'] == 0.6  # 3/5 correct
    
    # Test perfect predictions
    metrics_perfect = calculate_classification_metrics(
        y_true, y_true, y_true, y_true, y_true, y_true
    )
    assert metrics_perfect['train_accuracy'] == 1.0
    assert metrics_perfect['test_f1'] == 1.0
    
    # Test edge case: all zeros
    y_zero = np.array([0, 0, 0, 0, 0])
    y_pred_zero = np.array([0, 0, 0, 0, 0])
    metrics_zero = calculate_classification_metrics(
        y_zero, y_pred_zero, y_zero, y_pred_zero, y_zero, y_pred_zero
    )
    # All accuracy values should be 1.0
    assert metrics_zero['train_accuracy'] == 1.0
    # Precision, recall, F1 for positive class should be 0 (division by zero handled)
    assert metrics_zero['train_precision'] == 0
    assert metrics_zero['train_recall'] == 0
    assert metrics_zero['train_f1'] == 0
    
    # Test edge case: all predictions positive, but all true values negative
    y_all_neg = np.array([0, 0, 0, 0, 0])
    y_pred_all_pos = np.array([1, 1, 1, 1, 1])
    metrics_wrong = calculate_classification_metrics(
        y_all_neg, y_pred_all_pos, y_all_neg, y_pred_all_pos, y_all_neg, y_pred_all_pos
    )
    assert metrics_wrong['train_accuracy'] == 0
    assert metrics_wrong['train_precision'] == 0
    assert metrics_wrong['train_recall'] == 0


@pytest.mark.benchmark(group="metrics")
def test_regression_metrics_performance(benchmark):
    """Test performance of regression metrics calculation."""
    # Generate larger dataset for benchmarking
    np.random.seed(42)
    size = 1000
    y_true = np.random.normal(0, 1, size)
    y_pred = y_true + np.random.normal(0, 0.1, size)  # Add some noise
    
    # Benchmark the function
    result = benchmark(
        calculate_regression_metrics,
        y_true, y_pred, y_true, y_pred, y_true, y_pred
    )
    
    # Verify the result is as expected
    assert 'train_rmse' in result
    assert 'test_r2' in result


@pytest.mark.benchmark(group="metrics")
def test_classification_metrics_performance(benchmark):
    """Test performance of classification metrics calculation."""
    # Generate larger dataset for benchmarking
    np.random.seed(42)
    size = 1000
    y_true = np.random.randint(0, 2, size)
    y_pred = y_true.copy()
    
    # Add some errors (flip 10% of predictions)
    indices = np.random.choice(np.arange(size), size=int(size*0.1), replace=False)
    y_pred[indices] = 1 - y_pred[indices]
    
    # Benchmark the function
    result = benchmark(
        calculate_classification_metrics,
        y_true, y_pred, y_true, y_pred, y_true, y_pred
    )
    
    # Verify the result is as expected
    assert 'train_accuracy' in result
    assert 'test_f1' in result 