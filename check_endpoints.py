#!/usr/bin/env python3
"""
Check the FastAPI endpoints in the application.
"""
from src.app.main import app


def main():
    """Print information about available endpoints."""
    print("FastAPI Endpoints:")
    
    for route in app.routes:
        path = getattr(route, "path", "Unknown")
        methods = getattr(route, "methods", set())
        print(f"- {path} {methods}")


if __name__ == "__main__":
    main() 