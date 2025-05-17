#!/usr/bin/env python3
"""
A simple script to verify the endpoints in the FastAPI application.
"""
import sys
from pathlib import Path

from src.app.main import app

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Print information about available endpoints."""
    print("Checking FastAPI endpoints...")

    # Get all routes defined in the FastAPI app
    routes = app.routes

    # Print information about each route
    for route in routes:
        path = getattr(route, "path", "Unknown")
        methods = getattr(route, "methods", ["Unknown"])
        name = getattr(route, "name", "Unknown")

        print(f"Path: {path}")
        print(f"Methods: {methods}")
        print(f"Name: {name}")
        print("-" * 40)

    # Check for specific endpoints
    paths = [route.path for route in routes]
    required_endpoints = [
        "/add_data",
        "/retrain",
        "/metrics/{experiment_id}",
        "/deploy/{experiment_id}",
    ]

    for endpoint in required_endpoints:
        if endpoint in paths:
            print(f"✅ Found endpoint: {endpoint}")
        else:
            print(f"❌ Missing endpoint: {endpoint}")


if __name__ == "__main__":
    main()
