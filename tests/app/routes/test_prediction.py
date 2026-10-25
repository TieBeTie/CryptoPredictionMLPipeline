"""
Tests for the prediction endpoints.
"""
import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException

from src.app.routes.prediction import forward, forward_batch, process_batch_item


@pytest.mark.asyncio
async def test_forward_successful(prediction_data):
    """Test that the forward endpoint returns a valid prediction."""
    # Setup mock session
    mock_session = pytest.MockONNXSession()
    
    # Test successful prediction
    result = await forward(prediction_data, mock_session)
    
    # Check result structure
    assert "prediction" in result
    assert "features_used" in result
    assert isinstance(result["prediction"], float)
    assert result["features_used"] == prediction_data["features"]


@pytest.mark.asyncio
async def test_forward_error_handling():
    """Test error handling in the forward endpoint."""
    # Setup mock session that raises an exception
    mock_session = pytest.MockONNXSession()
    
    # Create a request with missing features
    request = {"features_metadata": {"source": "test"}}
    
    # Test for exception being raised
    with pytest.raises(HTTPException) as excinfo:
        await forward(request, mock_session)
    
    # Check exception details
    assert excinfo.value.status_code == 400
    assert "Prediction failed" in excinfo.value.detail


@pytest.mark.asyncio
async def test_process_batch_item_success():
    """Test processing a single batch item."""
    # Setup mock session
    mock_session = pytest.MockONNXSession()
    
    # Test item
    test_item = {
        "feature1": 0.5, "feature2": 1.0, "feature3": -0.5,
        "feature4": 0.1, "feature5": -0.1, "feature6": 0.2,
        "feature7": -0.2, "feature8": 0.3, "feature9": -0.3,
        "feature10": 0.4
    }
    
    # Process item
    result = await process_batch_item(test_item, mock_session)
    
    # Check result structure
    assert "prediction" in result
    assert "features_used" in result
    assert isinstance(result["prediction"], float)
    assert result["features_used"] == test_item


@pytest.mark.asyncio
async def test_process_batch_item_error():
    """Test error handling in batch item processing."""
    # Setup mock session
    mock_session = pytest.MockONNXSession()
    
    # Test item with wrong type (will cause numpy error)
    test_item = {
        "feature1": "invalid", "feature2": 1.0
    }
    
    # Process item (should not raise exception but return error)
    result = await process_batch_item(test_item, mock_session)
    
    # Check error is captured in result
    assert "error" in result
    assert "features_used" in result
    assert result["features_used"] == test_item


@pytest.mark.asyncio
async def test_forward_batch_streaming(batch_data):
    """Test that the batch endpoint streams results properly."""
    # Setup mock session
    mock_session = pytest.MockONNXSession()
    
    # Mock background tasks
    mock_background_tasks = pytest.MockBackgroundTasks = type(
        "MockBackgroundTasks", (), {"add_task": lambda self, task: None}
    )()
    
    # Call batch endpoint
    stream_response = await forward_batch(batch_data, mock_background_tasks, mock_session)
    
    # Check response type
    assert stream_response.media_type == "application/x-ndjson"
    
    # Helper to consume stream
    async def consume_stream():
        results = []
        async for line in stream_response.body_iterator:
            results.append(json.loads(line.decode()))
        return results
    
    # Get all responses from stream
    results = await consume_stream()
    
    # Check that we got 3 results (matching input batch size)
    assert len(results) == 3
    
    # Check structure of each result
    for i, result in enumerate(results):
        assert "prediction" in result
        assert "features_used" in result
        assert isinstance(result["prediction"], float)
        assert result["features_used"] == batch_data["instances"][i]


# Integration tests with FastAPI TestClient
def test_forward_endpoint_integration(client, prediction_data):
    """Test the forward endpoint through the FastAPI app."""
    response = client.post("/forward", json=prediction_data)
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "prediction" in data
    assert "features_used" in data
    assert data["features_used"] == prediction_data["features"]


def test_forward_invalid_request(client):
    """Test forward endpoint with invalid data."""
    # Empty request body
    response = client.post("/forward", json={})
    assert response.status_code in (400, 422)  # Validation error
    
    # Invalid feature type
    response = client.post("/forward", json={
        "features": {"feature1": "invalid"}
    })
    assert response.status_code in (400, 422)  # Validation error


def test_forward_batch_endpoint_integration(client, batch_data):
    """Test the batch endpoint through the FastAPI app."""
    response = client.post("/forward_batch", json=batch_data)
    assert response.status_code == 200
    
    # Check that we get a streaming response
    assert response.headers["content-type"] == "application/x-ndjson"
    
    # Parse each line as JSON
    results = [json.loads(line) for line in response.text.strip().split("\n")]
    
    # Check that we got 3 results (matching input batch size)
    assert len(results) == 3
    
    # Check structure of each result
    for i, result in enumerate(results):
        assert "prediction" in result
        assert "features_used" in result
        assert result["features_used"] == batch_data["instances"][i]


@pytest.mark.parametrize(
    "invalid_data,expected_status",
    [
        ({}, 422),  # Empty request
        ({"instances": []}, 200),  # Empty instances (valid but results in empty response)
        ({"instances": [{"invalid": "data"}]}, 200),  # Invalid instance but should not crash
    ],
)
def test_batch_endpoint_edge_cases(client, invalid_data, expected_status):
    """Test batch endpoint with edge cases."""
    response = client.post("/forward_batch", json=invalid_data)
    assert response.status_code == expected_status


@pytest.mark.benchmark(group="prediction")
def test_forward_performance(client, prediction_data, benchmark):
    """Test forward endpoint performance."""
    def make_prediction():
        return client.post("/forward", json=prediction_data)
    
    # Run benchmark
    result = benchmark(make_prediction)
    
    # Verify result
    assert result.status_code == 200
    data = result.json()
    assert "prediction" in data
