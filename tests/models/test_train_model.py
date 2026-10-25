import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.models.train_model import (
    FIGURES_PATH,
    calculate_classification_metrics,
    calculate_regression_metrics,
    create_model_pipeline,
    export_model_to_onnx,
    get_git_commit_hash,
    load_feature_data,
    main,
    plot_confusion_matrix,
    plot_feature_importance,
    prepare_data,
    setup_directories,
    train_and_evaluate_model,
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
    with pytest.raises(ValueError):  # Should raise error when target col doesn't exist
        prepare_data(missing_target_df, target_type="classification")


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


def test_regression_metrics_edge_cases():
    """Test regression metrics with edge cases."""
    # Perfect predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    metrics_perfect = calculate_regression_metrics(
        y_true, y_perfect, y_true, y_perfect, y_true, y_perfect
    )
    
    assert metrics_perfect['train_rmse'] == 0.0
    assert metrics_perfect['test_r2'] == 1.0
    
    # Very bad predictions (constant value)
    y_const = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    
    metrics_bad = calculate_regression_metrics(
        y_true, y_const, y_true, y_const, y_true, y_const
    )
    
    assert metrics_bad['train_r2'] < 0  # R2 can be negative for bad models
    
    # Single value
    y_single_true = np.array([1.0])
    y_single_pred = np.array([1.1])
    
    metrics_single = calculate_regression_metrics(
        y_single_true, y_single_pred, 
        y_single_true, y_single_pred,
        y_single_true, y_single_pred
    )
    
    assert 'train_rmse' in metrics_single
    assert metrics_single['train_rmse'] == 0.1


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


def test_classification_metrics_edge_cases():
    """Test classification metrics with edge cases."""
    # Perfect predictions
    y_true = np.array([0, 1, 0, 1, 0])
    y_perfect = np.array([0, 1, 0, 1, 0])
    
    metrics_perfect = calculate_classification_metrics(
        y_true, y_perfect, y_true, y_perfect, y_true, y_perfect
    )
    
    assert metrics_perfect['train_accuracy'] == 1.0
    assert metrics_perfect['test_precision'] == 1.0
    assert metrics_perfect['test_recall'] == 1.0
    
    # All zeros prediction
    y_all_zeros = np.array([0, 0, 0, 0, 0])
    
    metrics_zeros = calculate_classification_metrics(
        y_true, y_all_zeros, y_true, y_all_zeros, y_true, y_all_zeros
    )
    
    # Testing zero_division=0 parameter is working
    assert metrics_zeros['train_precision'] == 0.0
    assert metrics_zeros['test_recall'] == 0.0
    
    # Single value
    y_single_true = np.array([1])
    y_single_pred = np.array([1])
    
    metrics_single = calculate_classification_metrics(
        y_single_true, y_single_pred,
        y_single_true, y_single_pred,
        y_single_true, y_single_pred
    )
    
    assert metrics_single['train_accuracy'] == 1.0


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
def test_plot_feature_importance(mock_figure, mock_savefig, monkeypatch, temp_dirs):
    """Test feature importance plotting."""
    # Setup
    monkeypatch.setattr('src.models.train_model.FIGURES_PATH', temp_dirs['reports'])
    
    # Create test data
    feature_names = ['feature1', 'feature2', 'feature3']
    importances = np.array([0.5, 0.3, 0.2])
    
    # Call function
    result = plot_feature_importance(importances, feature_names)
    
    # Check results
    assert mock_figure.called
    assert mock_savefig.called
    assert result.parent == temp_dirs['reports']
    assert 'model_feature_importance_' in result.name
    assert result.suffix == '.png'


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
def test_plot_confusion_matrix(mock_figure, mock_savefig, monkeypatch, temp_dirs):
    """Test confusion matrix plotting."""
    # Setup
    monkeypatch.setattr('src.models.train_model.FIGURES_PATH', temp_dirs['reports'])
    
    # Create test data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    
    # Call function
    result = plot_confusion_matrix(y_true, y_pred)
    
    # Check results
    assert mock_figure.called
    assert mock_savefig.called
    assert result.parent == temp_dirs['reports']
    assert 'confusion_matrix_' in result.name
    assert result.suffix == '.png'


def test_get_git_commit_hash():
    """Test git commit hash retrieval."""
    # Test successful case
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = 'abcdef1234567890\n'
        commit_hash = get_git_commit_hash()
        assert commit_hash == 'abcdef1234567890'
        mock_run.assert_called_once()
    
    # Test failure case - subprocess error
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = Exception("Command failed")
        commit_hash = get_git_commit_hash()
        assert commit_hash == 'unknown-commit-hash'


@patch('src.models.train_model.convert_sklearn')
def test_export_model_to_onnx(mock_convert, monkeypatch, temp_dirs):
    """Test ONNX model export."""
    # Setup
    model_path = temp_dirs['models'] / 'model.onnx'
    monkeypatch.setattr('src.models.train_model.ONNX_MODEL_PATH', model_path)
    
    # Create mocks
    mock_model = MagicMock()
    mock_onnx_model = MagicMock()
    mock_convert.return_value = mock_onnx_model
    mock_onnx_model.metadata_props = []
    
    # Mock open and write
    mock_open = MagicMock()
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Call function with mocked open
    with patch('builtins.open', mock_open):
        with patch('src.models.train_model.get_git_commit_hash', return_value='test-hash'):
            result = export_model_to_onnx(mock_model, 10, 'test-run-id')
    
    # Verify calls
    mock_convert.assert_called_once()
    mock_file.write.assert_called_once()
    assert result == model_path
    
    # Test error handling
    mock_convert.side_effect = Exception("Conversion error")
    with pytest.raises(Exception):
        with patch('builtins.open', mock_open):
            export_model_to_onnx(mock_model, 10, 'test-run-id')


@patch('mlflow.start_run')
@patch('src.models.train_model.Live')
def test_train_and_evaluate_model(mock_live_class, mock_start_run, monkeypatch):
    """Test train and evaluate model function with MLflow."""
    # Setup mocks
    mock_run = MagicMock()
    mock_run.info.run_id = 'test-run-id'
    mock_start_run.return_value.__enter__.return_value = mock_run
    
    mock_live = MagicMock()
    mock_live_class.return_value = mock_live
    
    # Mock metrics file
    metrics_path = Path('reports/metrics.json')
    monkeypatch.setattr('src.models.train_model.METRICS_PATH', metrics_path)
    
    # Create test data
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    X_valid = pd.DataFrame({'feature1': [7, 8], 'feature2': [9, 10]})
    y_valid = pd.Series([1, 0])
    X_test = pd.DataFrame({'feature1': [11, 12], 'feature2': [13, 14]})
    y_test = pd.Series([0, 1])
    
    # Create model pipeline mock
    model_pipeline = MagicMock()
    model_pipeline.predict.side_effect = lambda X: np.array([0, 1] * (len(X) // 2 + len(X) % 2))[:len(X)]
    model_pipeline.named_steps = {'model': MagicMock()}
    model_pipeline.named_steps['model'].feature_importances_ = np.array([0.7, 0.3])
    
    # Mock json dump
    mock_json_dump = MagicMock()
    monkeypatch.setattr('json.dump', mock_json_dump)
    
    # Mock open
    mock_open = MagicMock()
    monkeypatch.setattr('builtins.open', mock_open)
    
    # Test regression
    with patch('src.models.train_model.plot_feature_importance', return_value=Path('test_fig.png')):
        model, run_id, metrics = train_and_evaluate_model(
            X_train, X_valid, X_test,
            y_train, y_valid, y_test,
            model_pipeline, target_type='regression'
        )
    
    # Verify calls
    assert mock_start_run.called
    assert model_pipeline.fit.called
    assert model_pipeline.predict.call_count == 3
    assert mock_live.log_metric.called
    assert mock_open.called
    assert mock_json_dump.called
    assert run_id == 'test-run-id'
    assert isinstance(metrics, dict)
    assert 'train_rmse' in metrics
    
    # Test classification
    with patch('src.models.train_model.plot_feature_importance', return_value=Path('test_fig.png')):
        with patch('src.models.train_model.plot_confusion_matrix', return_value=Path('test_cm.png')):
            model, run_id, metrics = train_and_evaluate_model(
                X_train, X_valid, X_test,
                y_train, y_valid, y_test,
                model_pipeline, target_type='classification'
            )
    
    # Verify classification metrics
    assert 'train_accuracy' in metrics
    assert 'test_precision' in metrics


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


def test_performance_small_dataset():
    """Test performance on small dataset."""
    import time

    # Create small dataset
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=100),
        'symbol': 'BTC',
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target_next_day_return': np.random.normal(0, 1, 100),
        'target_next_day_direction': np.random.randint(0, 2, 100),
    })

    # Measure time for prepare_data
    start_time = time.time()
    X_train, X_valid, X_test, y_train, y_valid, y_test, features = prepare_data(df)
    prepare_time = time.time() - start_time
    
    # Verify performance
    assert prepare_time < 1.0, f"prepare_data took too long: {prepare_time:.2f} seconds"

    # Create model
    start_time = time.time()
    model = create_model_pipeline()
    create_time = time.time() - start_time
    
    # Verify performance
    assert create_time < 1.0, f"create_model_pipeline took too long: {create_time:.2f} seconds"
    
    # Measure metrics calculation
    y_true = np.random.random(100)
    y_pred = np.random.random(100)
    
    start_time = time.time()
    metrics = calculate_regression_metrics(y_true, y_pred, y_true, y_pred, y_true, y_pred)
    metrics_time = time.time() - start_time
    
    # Verify performance
    assert metrics_time < 1.0, f"calculate_regression_metrics took too long: {metrics_time:.2f} seconds"


@given(n_rows=st.integers(min_value=10, max_value=200))
@settings(max_examples=5)
def test_property_prepare_data(n_rows):
    """Property-based test for prepare_data with variable data sizes."""
    # Create sample data with variable size
    dates = pd.date_range(start='2020-01-01', periods=n_rows)
    
    df = pd.DataFrame({
        'date': dates,
        'symbol': ['BTC'] * n_rows,
        'feature1': np.random.uniform(0, 1, n_rows),
        'feature2': np.random.uniform(0, 1, n_rows),
        'target_next_day_return': np.random.uniform(0, 1, n_rows),
        'target_next_day_direction': np.random.randint(0, 2, n_rows),
    })
    
    # Test with default split ratio
    X_train, X_valid, X_test, y_train, y_valid, y_test, features = prepare_data(df)
    
    # Check all splits have data
    assert len(X_train) > 0
    assert len(X_valid) > 0
    assert len(X_test) > 0
    
    # Check total number of samples is preserved
    assert len(X_train) + len(X_valid) + len(X_test) == n_rows
