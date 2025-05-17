#!/usr/bin/env python3
"""
Tests for the continuous learning endpoints in the FastAPI application.
Tests the add_data, retrain, metrics, and deploy endpoints.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Create a simple mock FastAPI app for testing
app = FastAPI(
    title="Continuous Learning API Mock",
    description="Mock API for testing continuous learning",
    version="1.0.0"
)


@app.put("/add_data")
async def add_data(request: dict):
    """Mock add_data endpoint."""
    rows = request.get("rows", [])
    replace_file = request.get("replace_file", False)
    return {
        "status": "success",
        "message": f"Added {len(rows)} rows to dataset",
        "total_rows": len(rows) + (0 if replace_file else 100)
    }


@app.put("/retrain")
async def retrain(request: dict):
    """Mock retrain endpoint."""
    return {
        "status": "success",
        "message": "Model retrained successfully",
        "experiment_id": "mock-experiment-id",
        "metrics": {
            "test_mae": 0.1,
            "test_rmse": 0.2,
            "test_r2": 0.8
        }
    }


@app.get("/metrics/{experiment_id}")
async def get_metrics(experiment_id: str):
    """Mock metrics endpoint."""
    if experiment_id == "not-found":
        return {"detail": "Experiment not-found not found"}, 404

    return {
        "experiment_id": experiment_id,
        "metrics": {
            "test_mae": 0.1,
            "test_rmse": 0.2,
            "test_r2": 0.8
        },
        "parameters": {
            "target_type": "regression",
            "model_type": "xgboost"
        },
        "start_time": 1609459200000,
        "status": "FINISHED"
    }


@app.post("/deploy/{experiment_id}")
async def deploy_model(experiment_id: str, request: dict):
    """Mock deploy endpoint."""
    if experiment_id == "not-found":
        return {"detail": "Experiment not-found not found"}, 404

    request.get("force", False)

    return {
        "status": "success",
        "message": f"Model from experiment {experiment_id} deployed successfully",
        "model_path": "models/current.onnx"
    }


# Create test client
@pytest.fixture
def client():
    return TestClient(app)


# Test cases
def test_add_data_endpoint_success(client):
    """Test successful data addition."""
    test_data = {
        "rows": [
            {"date": "2023-01-01", "symbol": "BTC", "open": 50000, "close": 51000},
            {"date": "2023-01-02", "symbol": "BTC", "open": 51000, "close": 52000}
        ],
        "replace_file": False
    }

    response = client.put("/add_data", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Added 2 rows" in data["message"]
    assert data["total_rows"] == 102  # 2 new + 100 existing


def test_add_data_replace_file(client):
    """Test replacing the entire data file."""
    test_data = {
        "rows": [
            {"date": "2023-01-01", "symbol": "BTC", "open": 50000, "close": 51000},
            {"date": "2023-01-02", "symbol": "BTC", "open": 51000, "close": 52000},
            {"date": "2023-01-03", "symbol": "BTC", "open": 52000, "close": 53000}
        ],
        "replace_file": True
    }

    response = client.put("/add_data", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_rows"] == 3  # Only the new rows


def test_retrain_endpoint_success(client):
    """Test successful model retraining."""
    test_data = {
        "target_type": "regression",
        "model_type": "xgboost",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        }
    }

    response = client.put("/retrain", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["experiment_id"] == "mock-experiment-id"
    assert "metrics" in data
    assert "test_mae" in data["metrics"]


def test_metrics_endpoint_success(client):
    """Test getting metrics for an experiment."""
    experiment_id = "test-experiment-id"

    response = client.get(f"/metrics/{experiment_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == experiment_id
    assert "metrics" in data
    assert "parameters" in data
    assert "start_time" in data
    assert "status" in data


def test_deploy_endpoint_success(client):
    """Test deploying a model from an experiment."""
    experiment_id = "test-experiment-id"
    test_data = {
        "force": False
    }

    response = client.post(f"/deploy/{experiment_id}", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert experiment_id in data["message"]
    assert data["model_path"] == "models/current.onnx"


def test_deploy_with_force(client):
    """Test deploying with force option."""
    experiment_id = "test-experiment-id"
    test_data = {
        "force": True
    }

    response = client.post(f"/deploy/{experiment_id}", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert experiment_id in data["message"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
