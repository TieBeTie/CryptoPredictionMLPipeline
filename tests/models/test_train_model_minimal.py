import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock external modules
sys.modules['mlflow'] = MagicMock()
sys.modules['mlflow.sklearn'] = MagicMock()
sys.modules['xgboost'] = MagicMock()
sys.modules['skl2onnx'] = MagicMock()
sys.modules['skl2onnx.common'] = MagicMock()
sys.modules['skl2onnx.common.data_types'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()
sys.modules['seaborn'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['dvclive'] = MagicMock()

# Import the module only after mocking dependencies
from src.models.train_model import (
    setup_directories,
    load_feature_data,
    prepare_data,
    calculate_regression_metrics,
    calculate_classification_metrics,
    get_git_commit_hash,
    export_model_to_onnx,
    main,
)


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


def test_prepare_data_edge_cases():
    """Test prepare_data with edge cases and boundary values."""
    # Minimal dataset (just enough data for train/valid/test split)
    small_df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=10),
        'symbol': 'BTC',
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'target_next_day_return': np.random.rand(10),
        'target_next_day_direction': np.random.randint(0, 2, 10),
    })
    
    # Should work with minimal data
    result = prepare_data(small_df, test_size=0.2, valid_size=0.2)
    assert len(result) == 7  # Returns 7 values
    assert all(len(x) > 0 for x in result[:6])  # All splits should have data
    
    # Test custom split ratios
    result_custom = prepare_data(small_df, test_size=0.5, valid_size=0.1)
    assert len(result_custom[0]) < len(result[0])  # Training set should be smaller
    assert len(result_custom[2]) > len(result[2])  # Test set should be larger
    
    # Missing target column
    missing_target_df = small_df.drop(columns=['target_next_day_direction'])
    with pytest.raises(KeyError):  # Should raise error when target col doesn't exist
        prepare_data(missing_target_df, target_type="classification")


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


@patch('subprocess.run')
def test_get_git_commit_hash(mock_run):
    """Test getting git commit hash."""
    # Configure mock
    mock_run.return_value.stdout = "abcdef1234567890\n"
    mock_run.return_value.check = True

    # Test function
    result = get_git_commit_hash()
    assert result == "abcdef1234567890"

    # Test error case
    mock_run.side_effect = Exception("Git error")
    result = get_git_commit_hash()
    assert result == "unknown-commit-hash"


@patch('src.models.train_model.convert_sklearn')
def test_export_model_to_onnx(mock_convert, monkeypatch, temp_dirs):
    """Test ONNX export."""
    # Configure mocks
    mock_model = MagicMock()
    mock_onnx = MagicMock()
    mock_convert.return_value = mock_onnx
    
    # Setup metadata props
    mock_onnx.metadata_props = []
    
    # Monkeypatch constants
    monkeypatch.setattr('src.models.train_model.ONNX_MODEL_PATH', 
                       temp_dirs['models'] / 'model.onnx')
    
    # Mock open and file write
    with patch('builtins.open', create=True) as mock_open:
        # Call function
        result = export_model_to_onnx(mock_model, 10, "test-run-id")
        
        # Check results
        assert mock_convert.called
        assert mock_open.called
        assert result == temp_dirs['models'] / 'model.onnx'
        
    # Test error handling
    mock_convert.side_effect = Exception("ONNX conversion error")
    with pytest.raises(Exception):
        export_model_to_onnx(mock_model, 10, "test-run-id")


@patch('src.models.train_model.export_model_to_onnx')
@patch('src.models.train_model.train_and_evaluate_model')
@patch('src.models.train_model.create_model_pipeline')
@patch('src.models.train_model.prepare_data')
@patch('src.models.train_model.load_feature_data')
@patch('src.models.train_model.setup_directories')
def test_main(mock_setup, mock_load, mock_prepare, mock_create, 
              mock_train, mock_export):
    """Test the main function."""
    # Configure mocks
    mock_load.return_value = pd.DataFrame()
    mock_prepare.return_value = (
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        pd.Series(), pd.Series(), pd.Series(), ['feature1', 'feature2']
    )
    mock_create.return_value = MagicMock()
    mock_train.return_value = (MagicMock(), "test-run-id", {})
    mock_export.return_value = MagicMock()
    
    # Call function
    result = main()
    
    # Verify call flow
    mock_setup.assert_called_once()
    mock_load.assert_called_once()
    mock_prepare.assert_called_once()
    mock_create.assert_called_once()
    mock_train.assert_called_once()
    mock_export.assert_called_once()
    
    # Verify result structure
    assert len(result) == 3
    
    # Test error handling
    mock_load.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        main()


def test_main_with_different_target_types():
    """Test main function with different target types."""
    # Patch all dependencies
    with patch('src.models.train_model.setup_directories'), \
         patch('src.models.train_model.load_feature_data'), \
         patch('src.models.train_model.prepare_data') as mock_prepare, \
         patch('src.models.train_model.create_model_pipeline') as mock_create, \
         patch('src.models.train_model.train_and_evaluate_model'), \
         patch('src.models.train_model.export_model_to_onnx'):
        
        # Run with regression
        main(target_type="regression")
        mock_prepare.assert_called_with(mock_prepare.return_value, target_type="regression")
        mock_create.assert_called_with(model_type="xgboost", params=None, target_type="regression")
        
        # Run with classification
        main(target_type="classification")
        mock_prepare.assert_called_with(mock_prepare.return_value, target_type="classification")
        mock_create.assert_called_with(model_type="xgboost", params=None, target_type="classification")
        
        # Run with custom parameters
        custom_params = {"learning_rate": 0.01}
        main(params=custom_params)
        mock_create.assert_called_with(model_type="xgboost", params=custom_params, target_type="regression") 