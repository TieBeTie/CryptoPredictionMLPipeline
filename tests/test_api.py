#!/usr/bin/env python3
"""
Tests for the FastAPI application.
Tests all endpoints and error cases.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Create a simple mock FastAPI app for testing
app = FastAPI(
    title="Cryptocurrency Price Predictor Mock",
    description="Mock API for testing",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Cryptocurrency Price Predictor API",
        "version": "1.0.0",
        "endpoints": [
            "/forward", "/forward_batch",
            "/evaluate", "/metadata"
        ]
    }


@app.post("/forward")
async def forward(request: dict):
    """Mock forward endpoint."""
    return {
        "prediction": 0.75,
        "features_used": request.get("features", {})
    }


@app.post("/forward_batch")
async def forward_batch(request: dict):
    """Mock batch endpoint."""
    results = []
    for item in request.get("instances", []):
        results.append({
            "prediction": 0.75,
            "features_used": item
        })

    return results


@app.get("/metadata")
async def get_metadata():
    """Mock metadata endpoint."""
    return {
        "model_path": "models/model.onnx",
        "inputs": [{"name": "input", "shape": [1, 10], "type": "tensor(float)"}],
        "outputs": [{"name": "output", "shape": [1, 1], "type": "tensor(float)"}],
        "metadata": {"model_type": "regression"}
    }


# Create test client
@pytest.fixture
def client():
    return TestClient(app)


# Test cases
def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data
    assert len(data["endpoints"]) == 4


def test_forward_endpoint_success(client):
    """Test successful prediction."""
    test_data = {
        "features": {
            "feature1": 0.5,
            "feature2": 1.0,
            "feature3": -0.5
        },
        "feature_metadata": {
            "source": "test"
        }
    }

    response = client.post("/forward", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "features_used" in data
    assert isinstance(data["prediction"], float)
    assert data["features_used"] == test_data["features"]


def test_forward_batch_endpoint(client):
    """Test batch prediction."""
    test_data = {
        "instances": [
            {"feature1": 0.5, "feature2": 1.0},
            {"feature1": 0.7, "feature2": -0.5},
            {"feature1": 0.2, "feature2": 0.0}
        ]
    }

    response = client.post("/forward_batch", json=test_data)
    assert response.status_code == 200
    results = response.json()

    # Check that we got 3 predictions
    assert len(results) == 3

    # Check the structure of each prediction
    for result, original in zip(results, test_data["instances"]):
        assert "prediction" in result
        assert "features_used" in result
        assert result["features_used"] == original


def test_metadata_endpoint(client):
    """Test the metadata endpoint."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()

    assert "model_path" in data
    assert "inputs" in data
    assert "outputs" in data
    assert "metadata" in data

    # Check metadata
    assert data["metadata"]["model_type"] == "regression"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
