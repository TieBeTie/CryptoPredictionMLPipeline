#!/usr/bin/env python3
"""
Script to demonstrate Phase 4: Model training & tracking.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and print output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        cwd=cwd
    )
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)

    return result


def main():
    """Run Phase 4 tests."""
    print("=" * 80)
    print("PHASE 4: MODEL TRAINING & TRACKING")
    print("=" * 80)

    # Check if required directories exist
    for directory in ["models", "reports", "data/processed"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Run unit tests
    print("\nRunning unit tests for train_model.py...")
    run_command("pytest -xvs tests/models/test_train_model.py")

    # Run ONNX inference tests
    print("\nRunning ONNX inference tests...")
    run_command("pytest -xvs tests/models/test_onnx_inference.py")

    print("\n" + "=" * 80)
    print("Phase 4 completed successfully!")
    print("=" * 80)
    print("\nNote: The actual model training was skipped in this test run.")
    print("To run the full training pipeline, you would need to:")
    print("1. Start MLflow server: mlflow ui --port 5000")
    print("2. Train a regression model: python -m src.models.train_model --target-type regression")
    print("3. Train a classification model: python -m src.models.train_model "
          "--target-type classification")


if __name__ == "__main__":
    main()
