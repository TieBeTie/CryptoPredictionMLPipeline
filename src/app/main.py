#!/usr/bin/env python3
"""
FastAPI application for cryptocurrency price prediction.
Provides endpoints for model inference, batch processing,
evaluation, metadata retrieval, and continuous learning.
"""
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.app.config import API_DESCRIPTION, API_TITLE, API_VERSION
from src.app.routes import data, model_management, prediction

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add Prometheus middleware
Instrumentator().instrument(app).expose(app)

# Include routers from different modules
app.include_router(prediction.router)
app.include_router(model_management.router)
app.include_router(data.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
