"""
Routes for data management.
"""
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.app.config import PROCESSED_DATA_PATH
from src.app.models import DataAdditionRequest

router = APIRouter(prefix="", tags=["data"])


@router.put("/add_data", status_code=201)
async def add_data(request: DataAdditionRequest):
    """
    Add new labeled data rows to the dataset.

    Args:
        request: Data addition request with rows to add

    Returns:
        dict: Status message
    """
    try:
        # Check if data dir exists
        data_dir = PROCESSED_DATA_PATH.parent
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)

        # Convert rows to DataFrame
        new_data = pd.DataFrame(request.rows)

        # Validate required columns
        required_date_columns = ["date", "symbol"]
        for col in required_date_columns:
            if col not in new_data.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Required column '{col}' missing from data"
                )

        # If file exists and not replacing, append to it
        if PROCESSED_DATA_PATH.exists() and not request.replace_file:
            try:
                existing_data = pd.read_parquet(PROCESSED_DATA_PATH)
                # Ensure new data has the same columns as existing data
                missing_cols = set(existing_data.columns) - set(new_data.columns)
                if missing_cols:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing columns in new data: {missing_cols}"
                    )

                # Add new rows
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                combined_data.to_parquet(PROCESSED_DATA_PATH, index=False)
                return {
                    "status": "success",
                    "message": f"Added {len(new_data)} rows to dataset",
                    "total_rows": len(combined_data)
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error appending data: {str(e)}"
                )
        else:
            # Create new file or replace existing
            try:
                new_data.to_parquet(PROCESSED_DATA_PATH, index=False)
                return {
                    "status": "success",
                    "message": f"Created new dataset with {len(new_data)} rows",
                    "total_rows": len(new_data)
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error saving data: {str(e)}"
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data addition failed: {str(e)}"
        )
