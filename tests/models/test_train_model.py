import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.train_model import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    create_model_pipeline,
    load_feature_data,
    main,
    prepare_data,
    setup_directories,
)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock onnx modules before importing train_model
sys.modules['skl2onnx'] = MagicMock()
sys.modules['skl2onnx.common'] = MagicMock()
sys.modules['skl2onnx.common.data_types'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create dates
    dates = pd.date_range(start='2020-01-01', periods=100)

    # Create features
    np.random.seed(42)
    df = pd.DataFrame({
        'date': dates,
        'symbol': 'BTC',
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'target_next_day_close': np.random.normal(10000, 1000, 100),
        'target_next_day_return': np.random.normal(0.01, 0.05, 100),
        'target_next_day_direction': np.random.randint(0, 2, 100),
    })

    return df


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create directory structure
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        data_dir = tmp_path / "data/processed"

        models_dir.mkdir(parents=True)
        reports_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        yield {
            'root': tmp_path,
            'models': models_dir,
            'reports': reports_dir,
            'data': data_dir
        }


def test_setup_directories(monkeypatch):
    """Test that setup_directories creates necessary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Monkeypatch constants
        monkeypatch.setattr('src.models.train_model.MODELS_DIR', tmp_path / 'models')
        monkeypatch.setattr(
            'src.models.train_model.METRICS_PATH',
            tmp_path / 'reports/metrics.json')
        monkeypatch.setattr('src.models.train_model.FIGURES_PATH', tmp_path / 'reports/figures')

        # Run function
        setup_directories()

        # Check directories were created
        assert (tmp_path / 'models').exists()
        assert (tmp_path / 'reports').exists()
        assert (tmp_path / 'reports/figures').exists()


def test_load_feature_data(temp_dirs, sample_data, monkeypatch):
    """Test loading feature data."""
    # Save sample data
    data_path = temp_dirs['data'] / 'features.parquet'
    sample_data.to_parquet(data_path)

    # Monkeypatch constant
    monkeypatch.setattr('src.models.train_model.FEATURES_DATA_PATH', data_path)

    # Load data
    df = load_feature_data()

    # Verify data
    assert len(df) == len(sample_data)
    assert list(df.columns) == list(sample_data.columns)

    # Test error case: file not found
    with monkeypatch.context() as m:
        m.setattr('src.models.train_model.FEATURES_DATA_PATH', Path('nonexistent.parquet'))
        with pytest.raises(FileNotFoundError):
            load_feature_data()


def test_prepare_data(sample_data):
    """Test data preparation with different target types."""
    # Test regression
    X_train, X_valid, X_test, y_train, y_valid, y_test, features = prepare_data(
        sample_data, target_type="regression"
    )

    # Check splits and target
    assert len(X_train) + len(X_valid) + len(X_test) == len(sample_data)
    assert y_train.name == 'target_next_day_return'

    # Test classification
    X_train, X_valid, X_test, y_train, y_valid, y_test, features = prepare_data(
        sample_data, target_type="classification"
    )

    # Check splits and target
    assert len(X_train) + len(X_valid) + len(X_test) == len(sample_data)
    assert y_train.name == 'target_next_day_direction'

    # Test invalid target type
    with pytest.raises(ValueError):
        prepare_data(sample_data, target_type="invalid_type")

    # Test empty dataframe after dropping NaNs
    empty_df = sample_data.copy()
    empty_df.loc[:, 'feature1'] = np.nan
    with pytest.raises(ValueError):
        prepare_data(empty_df)


def test_create_model_pipeline():
    """Test model pipeline creation with different configurations."""
    # Test regression pipeline
    reg_pipeline = create_model_pipeline(target_type="regression")
    assert reg_pipeline.named_steps['model'].__class__.__name__ == 'XGBRegressor'

    # Test classification pipeline
    clf_pipeline = create_model_pipeline(target_type="classification")
    assert clf_pipeline.named_steps['model'].__class__.__name__ == 'XGBClassifier'

    # Test with custom params - use a parameter that doesn't conflict with defaults
    custom_params = {'gamma': 0.1, 'min_child_weight': 2}
    custom_pipeline = create_model_pipeline(params=custom_params)
    assert custom_pipeline.named_steps['model'].gamma == 0.1
    assert custom_pipeline.named_steps['model'].min_child_weight == 2

    # Test invalid model type
    with pytest.raises(ValueError):
        create_model_pipeline(model_type="invalid_model")


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


def test_classification_metrics():
    """Test classification metrics calculation."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = calculate_classification_metrics(
        y_true, y_pred, y_true, y_pred, y_true, y_pred
    )

    # Check metrics
    assert 'train_accuracy' in metrics
    assert 'valid_precision' in metrics
    assert 'test_recall' in metrics
    assert metrics['test_accuracy'] == 0.6


@patch('src.models.train_model.get_git_commit_hash', return_value='test-commit-hash')
@patch('src.models.train_model.export_model_to_onnx')
@patch('src.models.train_model.train_and_evaluate_model')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.set_experiment')
def test_main_function(
        mock_set_experiment,
        mock_tracking_uri,
        mock_train_eval,
        mock_export,
        mock_git,
        monkeypatch):
    """Test the main function with different branches."""
    # Mock return values
    mock_train_eval.return_value = (MagicMock(), 'test-run-id', {'test_metric': 0.5})
    mock_export.return_value = Path('models/model.onnx')

    # Create sample data
    sample_df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=100),
        'symbol': 'BTC',
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target_next_day_return': np.random.normal(0, 1, 100),
        'target_next_day_direction': np.random.randint(0, 2, 100),
    })

    # Mock file operations
    monkeypatch.setattr('src.models.train_model.load_feature_data', lambda: sample_df)
    monkeypatch.setattr('src.models.train_model.setup_directories', MagicMock())

    # Test regression
    result = main(target_type="regression")
    assert len(result) == 3
    assert mock_export.called
    mock_train_eval.assert_called_with(
        ANY, ANY, ANY, ANY, ANY, ANY, ANY, target_type="regression"
    )

    # Test classification
    result = main(target_type="classification")
    assert len(result) == 3
    mock_train_eval.assert_called_with(
        ANY, ANY, ANY, ANY, ANY, ANY, ANY, target_type="classification"
    )

    # Test custom params
    custom_params = {'gamma': 0.1}
    result = main(target_type="regression", params=custom_params)
    assert len(result) == 3

    # Test exception handling
    monkeypatch.setattr('src.models.train_model.load_feature_data',
                        MagicMock(side_effect=Exception("Test error")))
    with pytest.raises(Exception):
        main()
