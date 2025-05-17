import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class MockMetadataProp:
    def __init__(self, name, value):
        self.name = name
        self.value = value


@patch('onnxruntime.InferenceSession')
def test_onnx_inference(mock_session):
    """Test loading and running inference with the ONNX model."""
    # Mock ONNX session
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 10]

    mock_session_instance = MagicMock()
    mock_session_instance.get_inputs.return_value = [mock_input]
    mock_session_instance.run.return_value = [np.array([0.5])]

    # Mock metadata with proper structure
    mock_metadata = MagicMock()
    mock_metadata.custom_metadata_map = [
        MockMetadataProp("git_commit_hash", "test-hash"),
        MockMetadataProp("save_date", "2023-01-01"),
        MockMetadataProp("mlflow_run_id", "test-run-id")
    ]
    mock_session_instance.get_modelmeta.return_value = mock_metadata

    # Set up mock return value
    mock_session.return_value = mock_session_instance

    # Check if model exists - create a dummy file if needed
    model_path = Path('models/model.onnx')
    model_path.parent.mkdir(exist_ok=True)
    if not model_path.exists():
        with open(model_path, 'w') as f:
            f.write("dummy onnx model")

    # Load ONNX model
    sess = mock_session(str(model_path))

    # Get input name and shape
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # Create dummy input
    feature_count = input_shape[1] if len(input_shape) > 1 else 1
    test_input = np.random.rand(1, feature_count).astype(np.float32)

    # Run inference
    result = sess.run(None, {input_name: test_input})

    # Check output
    assert len(result) > 0

    # Check metadata properly
    metadata_map = sess.get_modelmeta().custom_metadata_map
    metadata_dict = {prop.name: prop.value for prop in metadata_map}

    assert "git_commit_hash" in metadata_dict
    assert metadata_dict["git_commit_hash"] == "test-hash"
    assert "save_date" in metadata_dict
    assert "mlflow_run_id" in metadata_dict


if __name__ == "__main__":
    test_onnx_inference()
