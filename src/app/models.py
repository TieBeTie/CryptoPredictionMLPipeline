"""
Pydantic models for request and response validation.
"""
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Model for single prediction request."""
    features: Dict[str, float] = Field(
        ..., description="Feature key-value pairs"
    )
    feature_metadata: Optional[Dict[str, str]] = Field(
        None, description="Optional metadata"
    )


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction request."""
    instances: List[Dict[str, float]] = Field(
        ..., description="List of feature dictionaries"
    )


class EvaluationResponse(BaseModel):
    """Model for evaluation response."""
    metrics: Dict[str, float] = Field(
        ..., description="Evaluation metrics"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores"
    )


class DataAdditionRequest(BaseModel):
    """Model for adding new data."""
    rows: List[Dict[str, Union[float, str]]] = Field(
        ..., description="List of data rows to add"
    )
    replace_file: bool = Field(
        False, description="Whether to replace the entire file"
    )


class RetrainRequest(BaseModel):
    """Model for retraining request."""
    target_type: str = Field(
        "regression", description="Type of prediction problem: regression or classification"
    )
    model_type: str = Field(
        "xgboost", description="Type of model to use"
    )
    parameters: Optional[Dict[str, Union[float, int, str]]] = Field(
        None, description="Model hyperparameters"
    )


class DeployRequest(BaseModel):
    """Model for deployment request."""
    force: bool = Field(
        False, description="Whether to force deployment even if model exists"
    )
