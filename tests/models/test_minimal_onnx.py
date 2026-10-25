"""Minimal test for ONNX model inference with mocked dependencies."""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock external modules
class MockONNXSession:
    """Mock ONNX inference session."""
    def __init__(self, model_path=None):
        self.model_path = model_path
        self._inputs = []
        self._outputs = []
        self._metadata = None
        
        # Create a default input
        input_info = MagicMock()
        input_info.name = "float_input"
        input_info.shape = [None, 10]
        input_info.type = "tensor(float)"
        self._inputs.append(input_info)
        
        # Create metadata
        self._metadata = MagicMock()
        self._metadata.custom_metadata_map = {
            "git_commit_hash": "abcdef1234567890",
            "save_date": "2023-01-01T00:00:00Z",
            "mlflow_run_id": "test-run-id",
            "description": "Cryptocurrency price prediction model"
        }
    
    def get_inputs(self):
        """Get model input information."""
        return self._inputs
    
    def get_outputs(self):
        """Get model output information."""
        return self._outputs
    
    def get_modelmeta(self):
        """Get model metadata."""
        return self._metadata
    
    def run(self, output_names, input_feed):
        """Run inference."""
        # For testing, just return a random prediction based on input size
        first_input = list(input_feed.values())[0]
        batch_size = first_input.shape[0]
        
        # Return random predictions
        return [np.random.rand(batch_size, 1).astype(np.float32)]


# Test functions for ONNX models
def load_onnx_model(model_path):
    """Load an ONNX model for inference."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    
    try:
        # In actual code, this would be: session = onnxruntime.InferenceSession(model_path)
        session = MockONNXSession(model_path)
        return session
    except Exception as e:
        raise RuntimeError(f"Error loading ONNX model: {e}")


def predict_with_onnx(session, features):
    """Make predictions using an ONNX model."""
    if not isinstance(features, np.ndarray):
        features = np.array(features, dtype=np.float32)
    
    # Ensure correct shape - add batch dimension if needed
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Ensure float32 data type
    if features.dtype != np.float32:
        features = features.astype(np.float32)
    
    # Get input details
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: features})
    return outputs[0]


def get_model_metadata(session):
    """Get metadata from ONNX model."""
    try:
        metadata = session.get_modelmeta().custom_metadata_map
        return metadata
    except Exception as e:
        raise RuntimeError(f"Error retrieving model metadata: {e}")


# Tests
def test_load_onnx_model():
    """Test loading an ONNX model."""
    # Create a temporary file to simulate the model
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        # Write some dummy content to the file
        tmp.write(b"dummy onnx content")
        tmp.flush()
        
        # Load the model
        session = load_onnx_model(tmp.name)
        
        # Check if session was created
        assert session is not None
        assert session.model_path == tmp.name
        
        # Check input/output info
        inputs = session.get_inputs()
        assert len(inputs) > 0
        assert inputs[0].name == "float_input"
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        load_onnx_model("nonexistent_model.onnx")


def test_predict_with_onnx():
    """Test making predictions with ONNX model."""
    # Create mock session
    session = MockONNXSession()
    
    # Test with different input formats
    # 1. 2D array (batch of samples)
    features_2d = np.random.rand(5, 10).astype(np.float32)
    predictions_2d = predict_with_onnx(session, features_2d)
    assert predictions_2d.shape[0] == 5  # Batch size preserved
    
    # 2. 1D array (single sample)
    features_1d = np.random.rand(10).astype(np.float32)
    predictions_1d = predict_with_onnx(session, features_1d)
    assert predictions_1d.shape[0] == 1  # Single prediction
    
    # 3. List input (should be converted to array)
    features_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predictions_list = predict_with_onnx(session, features_list)
    assert predictions_list.shape[0] == 1
    
    # 4. Different data type (should be converted to float32)
    features_int = np.ones((2, 10), dtype=np.int32)
    predictions_int = predict_with_onnx(session, features_int)
    assert predictions_int.shape[0] == 2


def test_get_model_metadata():
    """Test retrieving metadata from ONNX model."""
    # Create mock session
    session = MockONNXSession()
    
    # Get metadata
    metadata = get_model_metadata(session)
    
    # Check key fields
    assert "git_commit_hash" in metadata
    assert metadata["git_commit_hash"] == "abcdef1234567890"
    assert "save_date" in metadata
    assert "mlflow_run_id" in metadata
    assert "description" in metadata


@pytest.mark.benchmark(group="onnx")
def test_onnx_single_prediction_performance(benchmark):
    """Test performance of ONNX model prediction with a single sample."""
    session = MockONNXSession()
    
    # Generate data for benchmarking
    np.random.seed(42)
    features = np.random.rand(10).astype(np.float32)
    
    # Benchmark single prediction
    result = benchmark(
        predict_with_onnx,
        session, features
    )
    
    # Verify the result
    assert result is not None
    assert result.shape[0] == 1


@pytest.mark.benchmark(group="onnx")
def test_onnx_batch_prediction_performance(benchmark):
    """Test performance of ONNX model prediction with batched samples."""
    session = MockONNXSession()
    
    # Generate batch data for benchmarking
    np.random.seed(42)
    batch_size = 100
    features = np.random.rand(batch_size, 10).astype(np.float32)
    
    # Benchmark batch prediction
    result = benchmark(
        predict_with_onnx,
        session, features
    )
    
    # Verify batch result
    assert result is not None
    assert result.shape[0] == batch_size 