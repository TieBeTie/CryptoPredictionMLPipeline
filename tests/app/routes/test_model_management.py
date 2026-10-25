"""
Tests for the model management endpoints.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from src.app.routes.model_management import (
    deploy_model,
    evaluate,
    get_metadata,
    get_metrics,
    retrain,
)


@pytest.mark.asyncio
async def test_get_metadata():
    """Test metadata retrieval endpoint."""
    # Setup mock session
    mock_session = pytest.MockONNXSession()
    
    # Call the endpoint
    result = await get_metadata(mock_session)
    
    # Check result structure
    assert "model_path" in result
    assert "inputs" in result
    assert "outputs" in result
    assert "metadata" in result
    
    # Check metadata values
    assert result["metadata"]["git_commit_hash"] == "test-hash"
    assert result["metadata"]["mlflow_run_id"] == "test-run-id"


@pytest.mark.asyncio
@patch("mlflow.get_run")
async def test_get_metrics_success(mock_get_run, mock_mlflow_run):
    """Test successful metrics retrieval."""
    # Configure mock
    mock_get_run.return_value = mock_mlflow_run
    
    # Call the endpoint
    result = await get_metrics("test-run-id")
    
    # Check result structure
    assert result["experiment_id"] == "test-run-id"
    assert "metrics" in result
    assert "parameters" in result
    assert "start_time" in result
    assert "status" in result
    
    # Check metrics values
    assert result["metrics"]["test_mae"] == 0.1
    assert result["metrics"]["test_rmse"] == 0.2
    assert result["metrics"]["test_r2"] == 0.8


@pytest.mark.asyncio
@patch("mlflow.get_run")
async def test_get_metrics_not_found(mock_get_run):
    """Test metrics retrieval with non-existent experiment."""
    # Configure mock to return None (experiment not found)
    mock_get_run.return_value = None
    
    # Test for exception being raised
    with pytest.raises(HTTPException) as excinfo:
        await get_metrics("nonexistent-id")
    
    # Check exception details
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.detail


@pytest.mark.asyncio
@patch("src.models.train_model.evaluate_model")
async def test_evaluate_success(mock_evaluate_model, tmp_path):
    """Test successful model evaluation with uploaded CSV file."""
    # Configure mock
    mock_evaluate_model.return_value = (
        {"accuracy": 0.85, "precision": 0.8, "recall": 0.9, "f1": 0.85},
        {"feature1": 0.5, "feature2": 0.3}
    )
    
    # Create a test CSV file
    csv_data = """date,symbol,feature1,feature2,target
2023-01-01,BTC,0.5,1.0,0.02
2023-01-02,BTC,0.7,-0.5,0.03
2023-01-03,BTC,0.2,0.0,0.01
"""
    temp_file = tmp_path / "test_data.csv"
    temp_file.write_text(csv_data)
    
    # Create a mock file upload
    class MockFile:
        def __init__(self, path):
            self.filename = path.name
            self.file = open(path, "rb")
        
        async def read(self):
            self.file.seek(0)
            return self.file.read()
        
        def close(self):
            self.file.close()
    
    mock_file = MockFile(temp_file)
    
    # Call the endpoint
    result = await evaluate(mock_file)
    mock_file.close()
    
    # Check result structure
    assert "metrics" in result
    assert "feature_importance" in result
    
    # Check metrics values
    assert result["metrics"]["accuracy"] == 0.85
    assert result["metrics"]["f1"] == 0.85
    
    # Check feature importance
    assert result["feature_importance"]["feature1"] == 0.5
    assert result["feature_importance"]["feature2"] == 0.3


@pytest.mark.asyncio
async def test_evaluate_invalid_file(tmp_path):
    """Test model evaluation with invalid file type."""
    # Create a test text file (not CSV)
    temp_file = tmp_path / "test_data.txt"
    temp_file.write_text("This is not a CSV file")
    
    # Create a mock file upload
    class MockFile:
        def __init__(self, path):
            self.filename = path.name
            self.file = open(path, "rb")
        
        async def read(self):
            self.file.seek(0)
            return self.file.read()
        
        def close(self):
            self.file.close()
    
    mock_file = MockFile(temp_file)
    
    # Test for exception being raised
    with pytest.raises(HTTPException) as excinfo:
        await evaluate(mock_file)
    
    mock_file.close()
    
    # Check exception details
    assert excinfo.value.status_code == 400
    assert "Unsupported file format" in excinfo.value.detail


@pytest.mark.asyncio
@patch("mlflow.start_run")
@patch("src.models.train_model.main")
async def test_retrain_success(mock_train_model, mock_start_run, retrain_request):
    """Test successful model retraining."""
    # Configure mocks
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_start_run.return_value.__enter__.return_value = mock_run
    
    mock_train_model.return_value = (
        MagicMock(),  # trained model
        {"test_mae": 0.1, "test_rmse": 0.2, "test_r2": 0.8},  # metrics
        Path("models/model.onnx")  # onnx path
    )
    
    # Mock Response object
    response = Response()
    
    # Call the endpoint
    result = await retrain(retrain_request, MagicMock(), response)
    
    # Check result structure
    assert result["status"] == "success"
    assert result["message"] == "Model retrained successfully"
    assert result["experiment_id"] == "test-run-id"
    assert "metrics" in result
    
    # Check metrics values
    assert result["metrics"]["test_mae"] == 0.1
    assert result["metrics"]["test_rmse"] == 0.2
    assert result["metrics"]["test_r2"] == 0.8
    
    # Check that the train_model function was called with the right parameters
    mock_train_model.assert_called_once_with(
        target_type=retrain_request.target_type,
        model_type=retrain_request.model_type,
        params=retrain_request.parameters
    )


@pytest.mark.asyncio
@patch("src.app.dependencies.reset_session")
@patch("pathlib.Path.symlink_to")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.unlink")
@patch("mlflow.get_run")
async def test_deploy_model_success(
    mock_get_run, mock_unlink, mock_exists, mock_symlink, mock_reset, mock_mlflow_run, tmp_path
):
    """Test successful model deployment."""
    # Configure mocks
    mock_get_run.return_value = mock_mlflow_run
    mock_exists.return_value = True  # Model file exists
    
    # Create request
    deploy_request = MagicMock()
    deploy_request.force = False
    
    # Mock paths
    with patch("src.app.config.MODEL_PATH", tmp_path / "models/model.onnx"), \
         patch("src.app.config.CURRENT_MODEL_PATH", tmp_path / "models/current.onnx"):
        
        # Create model file
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        (tmp_path / "models/model.onnx").touch()
        
        # Call the endpoint
        result = await deploy_model("test-run-id", deploy_request)
    
    # Check result structure
    assert result["status"] == "success"
    assert "test-run-id" in result["message"]
    assert "model_path" in result
    
    # Check that symlink was created
    mock_symlink.assert_called_once()
    
    # Check that session was reset
    mock_reset.assert_called_once()


@pytest.mark.asyncio
@patch("mlflow.get_run")
async def test_deploy_model_experiment_not_found(mock_get_run):
    """Test model deployment with non-existent experiment."""
    # Configure mock to return None (experiment not found)
    mock_get_run.return_value = None
    
    # Create request
    deploy_request = MagicMock()
    deploy_request.force = False
    
    # Test for exception being raised
    with pytest.raises(HTTPException) as excinfo:
        await deploy_model("nonexistent-id", deploy_request)
    
    # Check exception details
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.detail


@pytest.mark.asyncio
@patch("pathlib.Path.exists")
@patch("mlflow.get_run")
async def test_deploy_model_already_deployed(mock_get_run, mock_exists, mock_mlflow_run):
    """Test model deployment when model is already deployed."""
    # Configure mocks
    mock_get_run.return_value = mock_mlflow_run
    
    # First call: check if model exists
    # Second call: check if current.onnx exists
    mock_exists.side_effect = [True, True]
    
    # Create request with force=False
    deploy_request = MagicMock()
    deploy_request.force = False
    
    # Test for exception being raised
    with pytest.raises(HTTPException) as excinfo:
        await deploy_model("test-run-id", deploy_request)
    
    # Check exception details
    assert excinfo.value.status_code == 400
    assert "Model already deployed" in excinfo.value.detail


# Integration tests with FastAPI TestClient
def test_metadata_endpoint_integration(client):
    """Test the metadata endpoint through the FastAPI app."""
    response = client.get("/metadata")
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "model_path" in data
    assert "inputs" in data
    assert "outputs" in data
    assert "metadata" in data


@patch("mlflow.get_run")
def test_metrics_endpoint_integration(mock_get_run, client, mock_mlflow_run):
    """Test the metrics endpoint through the FastAPI app."""
    # Configure mock
    mock_get_run.return_value = mock_mlflow_run
    
    response = client.get("/metrics/test-run-id")
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert data["experiment_id"] == "test-run-id"
    assert "metrics" in data
    assert "parameters" in data


@patch("pathlib.Path.symlink_to")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.unlink")
@patch("mlflow.get_run")
def test_deploy_endpoint_integration(mock_get_run, mock_unlink, mock_exists, mock_symlink, client, mock_mlflow_run, tmp_path):
    """Test the deploy endpoint through the FastAPI app."""
    # Configure mocks
    mock_get_run.return_value = mock_mlflow_run
    mock_exists.return_value = True  # Model file exists
    
    # Reset session mock
    with patch("src.app.dependencies.reset_session"):
        response = client.post("/deploy/test-run-id", json={"force": True})
    
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert data["status"] == "success"
    assert "test-run-id" in data["message"]


@pytest.mark.benchmark(group="model_management")
def test_metadata_endpoint_performance(client, benchmark):
    """Test metadata endpoint performance."""
    # Run benchmark
    result = benchmark(client.get, "/metadata")
    
    # Verify result
    assert result.status_code == 200
    data = result.json()
    assert "metadata" in data 