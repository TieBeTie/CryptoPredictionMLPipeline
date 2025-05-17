#!/usr/bin/env python3
"""
Tests for Phase 8 - Docker and CI Configuration
Verifies Dockerfile and docker-compose.yml setup.
"""
import os
import unittest

import yaml


class TestDocker(unittest.TestCase):
    """Test Docker configuration."""

    def setUp(self):
        """Set up test environment."""
        # Check if docker-compose.yml exists
        self.assertTrue(os.path.exists("docker-compose.yml"),
                        "docker-compose.yml file doesn't exist")

        # Read docker-compose.yml
        with open("docker-compose.yml", "r") as f:
            self.docker_compose = yaml.safe_load(f)

        # Check if Dockerfile exists
        self.assertTrue(os.path.exists("Dockerfile"),
                        "Dockerfile doesn't exist")

        # Read Dockerfile
        with open("Dockerfile", "r") as f:
            self.dockerfile_content = f.read()

    def test_dockerfile_multistage(self):
        """Test that Dockerfile has multiple stages."""
        # Check for both builder and runtime stages
        self.assertIn("FROM", self.dockerfile_content)
        self.assertIn("AS builder", self.dockerfile_content)
        self.assertIn("COPY --from=builder", self.dockerfile_content)

    def test_dockerfile_healthcheck(self):
        """Test that Dockerfile has a healthcheck."""
        self.assertIn("HEALTHCHECK", self.dockerfile_content)

    def test_docker_compose_backend_service(self):
        """Test that docker-compose.yml has backend service properly configured."""
        services = self.docker_compose.get("services", {})
        self.assertIn("backend", services)

        backend = services.get("backend", {})
        self.assertIn("build", backend)
        self.assertIn("ports", backend)
        self.assertIn("8000:8000", backend.get("ports", []))
        self.assertIn("healthcheck", backend)

    def test_docker_compose_volumes(self):
        """Test that docker-compose.yml has volumes properly configured."""
        volumes = self.docker_compose.get("volumes", {})
        required_volumes = ["postgres_data", "prometheus_data", "grafana_data"]

        for volume in required_volumes:
            self.assertIn(volume, volumes)

    def test_ci_workflow_exists(self):
        """Test that GitHub Actions CI workflow exists."""
        workflow_path = ".github/workflows/ci.yml"
        self.assertTrue(os.path.exists(workflow_path),
                        "GitHub Actions workflow file doesn't exist")

        # Read workflow file
        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        # Check for required jobs
        jobs = workflow.get("jobs", {})
        self.assertIn("test", jobs)
        self.assertIn("dvc-pipeline", jobs)

        # Check for flake8
        test_job = jobs.get("test", {})
        steps = test_job.get("steps", [])

        flake8_step = next((step for step in steps if "Lint with flake8" in str(step)), None)
        self.assertIsNotNone(flake8_step, "flake8 step not found in workflow")

        # Check for pytest
        pytest_step = next((step for step in steps if "Run unit tests" in str(step)), None)
        self.assertIsNotNone(pytest_step, "pytest step not found in workflow")


if __name__ == "__main__":
    unittest.main()
