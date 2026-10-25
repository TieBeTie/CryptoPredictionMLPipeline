"""
Configure pytest fixtures for FastAPI tests.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Create a mock for the ONNX inference session
class MockONNXSession:
    """Mock for ONNXRuntime InferenceSession."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        
        # Create a mock input
        self.input = MagicMock()
        self.input.name = "float_input"
        self.input.shape = [None, 10]
        self.input.type = "tensor(float)"
        
        # Create a mock output
        self.output = MagicMock()
        self.output.name = "output"
        self.output.shape = [None, 1]
        self.output.type = "tensor(float)"
        
        # Create mock metadata
        self.metadata = MagicMock()
        self.metadata.custom_metadata_map = {
            "git_commit_hash": "test-hash",
            "save_date": "2023-01-01T00:00:00Z",
            "mlflow_run_id": "test-run-id",
            "description": "Cryptocurrency price prediction model"
        }
    
    def get_inputs(self):
        """Return mock inputs."""
        return [self.input]
    
    def get_outputs(self):
        """Return mock outputs."""
        return [self.output]
    
    def get_modelmeta(self):
        """Return mock metadata."""
        return self.metadata
    
    def run(self, output_names, input_feed):
        """Run mock inference."""
        # Get the shape of the input
        first_input = list(input_feed.values())[0]
        batch_size = first_input.shape[0]
        
        # Return random predictions
        return [np.random.rand(batch_size, 1).astype(np.float32)]


# Patch all necessary modules
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies."""
    # Create a patch for onnxruntime.InferenceSession
    with patch("onnxruntime.InferenceSession", MockONNXSession), \
         patch("src.app.dependencies.ort.InferenceSession", MockONNXSession), \
         patch("src.app.routes.model_management.ort.InferenceSession", MockONNXSession), \
         patch("src.app.dependencies.session", None), \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metrics"), \
         patch("mlflow.log_text"), \
         patch("mlflow.get_run"), \
         patch("subprocess.run"):
        yield


# Patch the model paths to use temporary files
@pytest.fixture(autouse=True)
def mock_model_paths(tmp_path):
    """Mock model paths to use temporary files."""
    model_path = tmp_path / "model.onnx"
    current_model_path = tmp_path / "current.onnx"
    
    # Create the model file
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "w") as f:
        f.write("mock onnx model")
    
    with patch("src.app.config.MODEL_PATH", model_path), \
         patch("src.app.config.CURRENT_MODEL_PATH", current_model_path):
        yield {
            "model_path": model_path,
            "current_model_path": current_model_path
        }


# Create a test client for the FastAPI app
@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    # Import here to avoid circular imports
    from src.app.main import app
    
    with TestClient(app) as test_client:
        yield test_client


# Create sample prediction data
@pytest.fixture
def prediction_data():
    """Sample data for prediction testing."""
    return {
        "features": {
            "feature1": 0.5,
            "feature2": 1.0,
            "feature3": -0.5,
            "feature4": 0.1,
            "feature5": -0.1,
            "feature6": 0.2,
            "feature7": -0.2,
            "feature8": 0.3,
            "feature9": -0.3,
            "feature10": 0.4
        },
        "feature_metadata": {
            "source": "test"
        }
    }


# Create sample batch prediction data
@pytest.fixture
def batch_data():
    """Sample data for batch prediction testing."""
    return {
        "instances": [
            {
                "feature1": 0.5, "feature2": 1.0, "feature3": -0.5,
                "feature4": 0.1, "feature5": -0.1, "feature6": 0.2,
                "feature7": -0.2, "feature8": 0.3, "feature9": -0.3,
                "feature10": 0.4
            },
            {
                "feature1": 0.7, "feature2": -0.5, "feature3": 0.8,
                "feature4": 0.2, "feature5": -0.3, "feature6": 0.4,
                "feature7": -0.6, "feature8": 0.1, "feature9": -0.8,
                "feature10": 0.9
            },
            {
                "feature1": 0.2, "feature2": 0.0, "feature3": 0.3,
                "feature4": 0.5, "feature5": -0.7, "feature6": 0.6,
                "feature7": -0.4, "feature8": 0.8, "feature9": -0.1,
                "feature10": 0.2
            }
        ]
    }


# Create sample data addition request
@pytest.fixture
def data_addition_request():
    """Sample data for data addition testing."""
    return {
        "rows": [
            {"date": "2023-01-01", "symbol": "BTC", "open": 50000, "close": 51000,
             "feature1": 0.5, "feature2": 1.0, "target_next_day_return": 0.02},
            {"date": "2023-01-02", "symbol": "BTC", "open": 51000, "close": 52000,
             "feature1": 0.7, "feature2": -0.5, "target_next_day_return": 0.02}
        ],
        "replace_file": False
    }


# Create sample retrain request
@pytest.fixture
def retrain_request():
    """Sample data for model retraining testing."""
    return {
        "target_type": "regression",
        "model_type": "xgboost",
        "parameters": {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1
        }
    }


# Mock MLflow run
@pytest.fixture
def mock_mlflow_run():
    """Mock MLflow run for testing."""
    class MockRun:
        def __init__(self):
            self.info = MagicMock()
            self.info.run_id = "test-run-id"
            self.info.start_time = 1609459200000
            self.info.status = "FINISHED"
            
            self.data = MagicMock()
            self.data.metrics = {
                "test_mae": 0.1,
                "test_rmse": 0.2,
                "test_r2": 0.8
            }
            self.data.params = {
                "target_type": "regression",
                "model_type": "xgboost"
            }
    
    return MockRun() 