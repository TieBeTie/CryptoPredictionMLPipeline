"""
Routes for prediction endpoints.
"""
import json
from typing import Dict, Union

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.app.dependencies import get_session
from src.app.models import BatchPredictionRequest, PredictionRequest

router = APIRouter(prefix="", tags=["prediction"])


@router.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Cryptocurrency Price Predictor API",
        "version": "1.0.0",
        "endpoints": [
            "/forward", "/forward_batch",
            "/evaluate", "/metadata",
            "/add_data", "/retrain",
            "/metrics/{id}", "/deploy/{id}"
        ]
    }


@router.post("/forward", response_model=Dict[str, Union[float, Dict]])
async def forward(request: PredictionRequest,
                  session: ort.InferenceSession = Depends(get_session)):
    """
    Make a single prediction.

    Args:
        request: Prediction request with features

    Returns:
        dict: Prediction result
    """
    try:
        # Convert features to numpy array
        features_list = list(request.features.values())
        input_name = session.get_inputs()[0].name
        X = np.array([features_list], dtype=np.float32)

        # Run inference
        outputs = session.run(None, {input_name: X})
        prediction = float(outputs[0][0])

        return {
            "prediction": prediction,
            "features_used": request.features
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


async def process_batch_item(item, session):
    """Process a single batch item."""
    try:
        features_list = list(item.values())
        input_name = session.get_inputs()[0].name
        X = np.array([features_list], dtype=np.float32)
        outputs = session.run(None, {input_name: X})
        return {
            "prediction": float(outputs[0][0]),
            "features_used": item
        }
    except Exception as e:
        return {"error": str(e), "features_used": item}


@router.post("/forward_batch")
async def forward_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    session: ort.InferenceSession = Depends(get_session)
):
    """
    Process batch predictions and stream results.

    Args:
        request: Batch prediction request with multiple instances

    Returns:
        StreamingResponse: Stream of prediction results
    """
    async def generate_predictions():
        """Generate predictions one by one and yield as JSON lines."""
        for item in request.instances:
            result = await process_batch_item(item, session)
            yield json.dumps(result) + "\n"

    return StreamingResponse(
        generate_predictions(),
        media_type="application/x-ndjson"
    )
