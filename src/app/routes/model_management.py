"""
Routes for model management (evaluation, retraining, deployment).
"""
import subprocess
import tempfile
import zipfile
from pathlib import Path

import mlflow
import onnxruntime as ort
import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from src.app.config import (
    CURRENT_MODEL_PATH,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
)
from src.app.dependencies import get_session, reset_session
from src.app.models import DeployRequest, EvaluationResponse, RetrainRequest
from src.models.train_model import evaluate_model
from src.models.train_model import main as train_model_main

router = APIRouter(prefix="", tags=["model-management"])


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(file: UploadFile = File(...)):
    """
    Evaluate model on uploaded dataset (CSV or ZIP containing CSV).

    Args:
        file: Uploaded file (CSV or ZIP)

    Returns:
        dict: Evaluation metrics
    """
    try:
        # Create temporary directory for file processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / file.filename

            # Save uploaded file
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Handle zip files
            if file.filename.endswith('.zip'):
                extract_dir = temp_path / "extracted"
                extract_dir.mkdir()
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Find CSV files
                csv_files = list(extract_dir.glob("*.csv"))
                if not csv_files:
                    raise HTTPException(
                        status_code=400,
                        detail="No CSV files found in ZIP archive"
                    )
                data_path = csv_files[0]
            else:
                if not file.filename.endswith('.csv'):
                    raise HTTPException(
                        status_code=400,
                        detail="Unsupported file format. Please upload CSV or ZIP"
                    )
                data_path = file_path

            # Load data
            try:
                eval_df = pd.read_csv(data_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading CSV: {str(e)}"
                )

            # Run evaluation
            model_to_use = CURRENT_MODEL_PATH if CURRENT_MODEL_PATH.exists() else MODEL_PATH
            metrics, feature_importance = evaluate_model(
                model_path=str(model_to_use),
                eval_data=eval_df
            )

            return {
                "metrics": metrics,
                "feature_importance": feature_importance
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/metadata")
async def get_metadata(session: ort.InferenceSession = Depends(get_session)):
    """
    Get metadata from ONNX model.

    Returns:
        dict: Model metadata
    """
    try:
        input_details = []
        for inp in session.get_inputs():
            input_details.append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type
            })

        output_details = []
        for out in session.get_outputs():
            output_details.append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type
            })

        # Get ONNX model metadata
        metadata = {}
        for k, v in session.get_modelmeta().custom_metadata_map.items():
            metadata[k] = v

        # Determine which model path is being used
        actual_model = CURRENT_MODEL_PATH if CURRENT_MODEL_PATH.exists() else MODEL_PATH

        return {
            "model_path": str(actual_model),
            "inputs": input_details,
            "outputs": output_details,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metadata: {str(e)}"
        )


@router.put("/retrain")
async def retrain(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    response: Response
):
    """
    Retrain the model using DVC pipeline and return experiment ID.

    Args:
        request: Retraining request parameters

    Returns:
        dict: Status message with experiment ID
    """
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log parameters to MLflow
            mlflow.log_params({
                "target_type": request.target_type,
                "model_type": request.model_type
            })

            if request.parameters:
                mlflow.log_params(request.parameters)

            try:
                # Run DVC training pipeline
                result = subprocess.run(
                    ["dvc", "repro", "train_model"],
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Log DVC output to MLflow
                mlflow.log_text(result.stdout, "dvc_output.txt")

                # Run the training directly in case DVC command failed silently
                trained_model, metrics, onnx_path = train_model_main(
                    target_type=request.target_type,
                    model_type=request.model_type,
                    params=request.parameters
                )

                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

                # Set status code to indicate success
                response.status_code = status.HTTP_200_OK

                return {
                    "status": "success",
                    "message": "Model retrained successfully",
                    "experiment_id": run_id,
                    "metrics": metrics
                }

            except subprocess.CalledProcessError as e:
                mlflow.log_text(f"Error: {str(e)}\nStdout: {e.stdout}\nStderr: {e.stderr}",
                                "error.txt")
                raise HTTPException(
                    status_code=500,
                    detail=f"DVC pipeline failed: {e.stderr}"
                )
            except Exception as e:
                mlflow.log_text(f"Error: {str(e)}", "error.txt")
                raise HTTPException(
                    status_code=500,
                    detail=f"Model training failed: {str(e)}"
                )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retraining failed: {str(e)}"
        )


@router.get("/metrics/{experiment_id}")
async def get_metrics(experiment_id: str):
    """
    Get metrics for a specific experiment.

    Args:
        experiment_id: MLflow experiment ID

    Returns:
        dict: Metrics for the experiment
    """
    try:
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Get experiment run
        run = mlflow.get_run(experiment_id)
        if not run:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id} not found"
            )

        # Return run metrics
        return {
            "experiment_id": experiment_id,
            "metrics": run.data.metrics,
            "parameters": run.data.params,
            "start_time": run.info.start_time,
            "status": run.info.status
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )


@router.post("/deploy/{experiment_id}")
async def deploy_model(experiment_id: str, request: DeployRequest):
    """
    Deploy a model from an experiment by creating a symlink.

    Args:
        experiment_id: MLflow experiment ID
        request: Deployment request with options

    Returns:
        dict: Status message
    """
    try:
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Get experiment run
        run = mlflow.get_run(experiment_id)
        if not run:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id} not found"
            )

        # Check if model exists
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found at {MODEL_PATH}"
            )

        # Check if current.onnx exists
        if CURRENT_MODEL_PATH.exists() and not request.force:
            raise HTTPException(
                status_code=400,
                detail="Model already deployed. Use force=true to overwrite"
            )

        # If current.onnx exists, remove it
        if CURRENT_MODEL_PATH.exists():
            CURRENT_MODEL_PATH.unlink()

        # Create symlink
        CURRENT_MODEL_PATH.symlink_to(MODEL_PATH)

        # Clear session to force reload with new model
        reset_session()

        return {
            "status": "success",
            "message": f"Model from experiment {experiment_id} deployed successfully",
            "model_path": str(CURRENT_MODEL_PATH)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Deployment failed: {str(e)}"
        )
