"""
Configuration settings for the FastAPI application.
"""
import os
from pathlib import Path

# Path to the ONNX model
MODEL_PATH = Path("models/model.onnx")
CURRENT_MODEL_PATH = Path("models/current.onnx")
METRICS_PATH = Path("reports/metrics.json")
PROCESSED_DATA_PATH = Path("data/processed/features.parquet")
RAW_DATA_PATH = Path("data/raw/crypto.parquet")

# MLflow settings
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT_NAME = "crypto_price_prediction"

# API settings
API_TITLE = "Cryptocurrency Price Predictor"
API_DESCRIPTION = "API for predicting cryptocurrency prices"
API_VERSION = "1.0.0"
