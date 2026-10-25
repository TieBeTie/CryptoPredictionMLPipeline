"""
Tests for the data management endpoints.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

from src.app.routes.data import add_data


@pytest.mark.asyncio
@patch("pandas.DataFrame.to_parquet")
@patch("pandas.read_parquet")
async def test_add_data_success(mock_read_parquet, mock_to_parquet, data_addition_request, tmp_path):
    """Test successful data addition."""
    # Setup mocks
    mock_read_parquet.return_value = pd.DataFrame({
        "date": ["2023-01-01"],
        "symbol": ["BTC"],
        "feature1": [0.5],
        "feature2": [1.0],
        "target_next_day_return": [0.01]
    })
    
    # Mock the Path and exists check
    with patch("src.app.config.PROCESSED_DATA_PATH", tmp_path / "data/processed/features.parquet"):
        # Make sure parent directory exists
        (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
        
        # Create an empty file
        (tmp_path / "data/processed/features.parquet").touch()
        
        # Call the endpoint
        result = await add_data(data_addition_request)
    
    # Check that read_parquet was called
    mock_read_parquet.assert_called_once()
    
    # Check that to_parquet was called with the combined data
    mock_to_parquet.assert_called_once()
    
    # Check result structure
    assert result["status"] == "success"
    assert "Added" in result["message"]
    assert "total_rows" in result
    assert result["total_rows"] > len(data_addition_request["rows"])


@pytest.mark.asyncio
@patch("pandas.DataFrame.to_parquet")
async def test_add_data_new_file(mock_to_parquet, data_addition_request, tmp_path):
    """Test creating a new data file."""
    # Set replace_file flag to True
    data_addition_request["replace_file"] = True
    
    # Mock the Path and exists check
    with patch("src.app.config.PROCESSED_DATA_PATH", tmp_path / "data/processed/features.parquet"):
        # Make sure parent directory exists
        (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
        
        # Call the endpoint
        result = await add_data(data_addition_request)
    
    # Check that to_parquet was called with the new data
    mock_to_parquet.assert_called_once()
    
    # Check result structure
    assert result["status"] == "success"
    assert "Created new dataset" in result["message"]
    assert result["total_rows"] == len(data_addition_request["rows"])


@pytest.mark.asyncio
async def test_add_data_missing_columns(data_addition_request):
    """Test adding data with missing required columns."""
    # Remove required column
    invalid_request = data_addition_request.copy()
    invalid_request["rows"] = [
        {k: v for k, v in row.items() if k != "date"}
        for row in data_addition_request["rows"]
    ]
    
    # Test for exception being raised
    with pytest.raises(HTTPException) as excinfo:
        await add_data(invalid_request)
    
    # Check exception details
    assert excinfo.value.status_code == 400
    assert "Required column 'date' missing" in excinfo.value.detail


@pytest.mark.asyncio
@patch("pandas.read_parquet")
async def test_add_data_schema_mismatch(mock_read_parquet, data_addition_request, tmp_path):
    """Test adding data with mismatched schema."""
    # Setup mock to return DataFrame with extra columns
    mock_read_parquet.return_value = pd.DataFrame({
        "date": ["2023-01-01"],
        "symbol": ["BTC"],
        "feature1": [0.5],
        "feature2": [1.0],
        "extra_column": [2.0],  # Extra column not in new data
        "target_next_day_return": [0.01]
    })
    
    # Mock the Path and exists check
    with patch("src.app.config.PROCESSED_DATA_PATH", tmp_path / "data/processed/features.parquet"):
        # Make sure parent directory exists
        (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
        
        # Create an empty file
        (tmp_path / "data/processed/features.parquet").touch()
        
        # Test for exception being raised
        with pytest.raises(HTTPException) as excinfo:
            await add_data(data_addition_request)
    
    # Check exception details
    assert excinfo.value.status_code == 400
    assert "Missing columns in new data" in excinfo.value.detail


@pytest.mark.asyncio
@patch("pandas.read_parquet")
async def test_add_data_error_handling(mock_read_parquet, data_addition_request, tmp_path):
    """Test error handling in data addition."""
    # Setup mock to raise an exception
    mock_read_parquet.side_effect = Exception("Test error")
    
    # Mock the Path and exists check
    with patch("src.app.config.PROCESSED_DATA_PATH", tmp_path / "data/processed/features.parquet"):
        # Make sure parent directory exists
        (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
        
        # Create an empty file
        (tmp_path / "data/processed/features.parquet").touch()
        
        # Test for exception being raised
        with pytest.raises(HTTPException) as excinfo:
            await add_data(data_addition_request)
    
    # Check exception details
    assert excinfo.value.status_code == 500
    assert "Error appending data" in excinfo.value.detail


# Integration tests with FastAPI TestClient
def test_add_data_endpoint_integration(client, data_addition_request, tmp_path):
    """Test the add_data endpoint through the FastAPI app."""
    # Mock file operations
    with patch("pandas.DataFrame.to_parquet"), \
         patch("pandas.read_parquet", return_value=pd.DataFrame()), \
         patch("src.app.config.PROCESSED_DATA_PATH", tmp_path / "data/processed/features.parquet"), \
         patch("pathlib.Path.exists", return_value=True):
        
        # Call the endpoint
        response = client.put("/add_data", json=data_addition_request)
        
        # Check response
        assert response.status_code == 201  # Created
        data = response.json()
        assert data["status"] == "success"
        assert "total_rows" in data


@pytest.mark.parametrize(
    "invalid_data,expected_status,expected_message",
    [
        ({"rows": []}, 201, "success"),  # Empty rows (valid but doesn't add anything)
        ({"rows": [{"missing_required": "values"}]}, 400, "Required column"),  # Missing required columns
        ({}, 422, None),  # No rows field
    ],
)
def test_add_data_edge_cases(client, invalid_data, expected_status, expected_message, tmp_path):
    """Test add_data endpoint with edge cases."""
    # Mock file operations
    with patch("pandas.DataFrame.to_parquet"), \
         patch("pandas.read_parquet", return_value=pd.DataFrame()), \
         patch("src.app.config.PROCESSED_DATA_PATH", tmp_path / "data/processed/features.parquet"), \
         patch("pathlib.Path.exists", return_value=True):
        
        # Call the endpoint
        response = client.put("/add_data", json=invalid_data)
        
        # Check response
        assert response.status_code == expected_status
        
        # If we expect a specific message, check for it
        if expected_message and response.status_code != 422:  # Skip validation errors
            assert expected_message in response.json().get("detail", "") 