"""
FastAPI dependencies for dependency injection.
"""
import onnxruntime as ort
from fastapi import HTTPException

from src.app.config import CURRENT_MODEL_PATH, MODEL_PATH

# Initialize session
session = None


def get_session():
    """
    Get or create an ONNX Runtime session.

    Returns:
        ort.InferenceSession: The ONNX Runtime session.
    """
    global session
    if session is None:
        model_to_use = CURRENT_MODEL_PATH if CURRENT_MODEL_PATH.exists() else MODEL_PATH

        if not model_to_use.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found at {model_to_use}"
            )
        session = ort.InferenceSession(str(model_to_use.resolve()))
    return session


def reset_session():
    """Reset the ONNX session to force model reload."""
    global session
    session = None
