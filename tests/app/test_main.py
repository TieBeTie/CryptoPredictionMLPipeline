"""
Tests for the main FastAPI application.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_fastapi_instrumentator import Instrumentator

from src.app.main import app


def test_app_title():
    """Test the FastAPI app title."""
    assert app.title == "Cryptocurrency Price Predictor"
    assert app.description == "API for predicting cryptocurrency prices"
    assert app.version == "1.0.0"


def test_routers_registered():
    """Test that all routers are registered with the app."""
    # Create a list of all route paths
    routes = [route.path for route in app.routes]
    
    # Check that expected endpoints are registered
    expected_endpoints = [
        "/",  # Root endpoint
        "/forward",  # Single prediction
        "/forward_batch",  # Batch prediction
        "/metadata",  # Model metadata
        "/add_data",  # Add new data
        "/retrain",  # Retrain model
        "/metrics/{experiment_id}",  # Get metrics
        "/deploy/{experiment_id}",  # Deploy model
        "/evaluate",  # Evaluate model
    ]
    
    for endpoint in expected_endpoints:
        assert endpoint in routes, f"Endpoint {endpoint} not found in routes"


def test_prometheus_middleware():
    """Test that Prometheus middleware is registered."""
    # Check if Prometheus endpoints are registered
    metrics_path = "/metrics"
    routes = [route.path for route in app.routes]
    assert metrics_path in routes, "Prometheus metrics endpoint not found" 