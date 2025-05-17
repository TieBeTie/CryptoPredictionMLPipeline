#!/usr/bin/env python3
"""
Tests for Phase 7 - Monitoring & Observability
Verifies Docker Compose services are configured properly.
"""
import os
import unittest

import yaml


class TestMonitoring(unittest.TestCase):
    """Test suite for monitoring and observability setup."""

    def setUp(self):
        """Set up test environment."""
        # Read docker-compose.yml
        with open("docker-compose.yml", "r") as f:
            self.docker_compose = yaml.safe_load(f)

        # Read Prometheus config
        with open("prometheus/prometheus.yml", "r") as f:
            self.prometheus_config = yaml.safe_load(f)

    def test_docker_compose_services(self):
        """Test that required services are defined in docker-compose.yml."""
        services = self.docker_compose.get("services", {})
        required_services = ["backend", "postgres", "prometheus", "grafana"]

        for service in required_services:
            self.assertIn(service, services, f"Service {service} not defined in docker-compose.yml")

    def test_volumes_defined(self):
        """Test that required volumes are defined in docker-compose.yml."""
        volumes = self.docker_compose.get("volumes", {})
        required_volumes = ["postgres_data", "prometheus_data", "grafana_data"]

        for volume in required_volumes:
            self.assertIn(volume, volumes, f"Volume {volume} not defined in docker-compose.yml")

    def test_ports_exposed(self):
        """Test that required ports are exposed in docker-compose.yml."""
        services = self.docker_compose.get("services", {})
        port_mappings = {
            "backend": "8000:8000",
            "prometheus": "9090:9090",
            "grafana": "3000:3000"
        }

        for service, port_mapping in port_mappings.items():
            self.assertIn(service, services, f"Service {service} not defined")
            ports = services[service].get("ports", [])
            self.assertIn(
                port_mapping,
                ports,
                f"Port mapping {port_mapping} not defined for {service}")

    def test_prometheus_config(self):
        """Test that Prometheus is configured to scrape the FastAPI backend."""
        scrape_configs = self.prometheus_config.get("scrape_configs", [])

        # Check for crypto_api job
        jobs = [config.get("job_name") for config in scrape_configs]
        self.assertIn("crypto_api", jobs, "Prometheus not configured to scrape crypto_api")

        # Check that it targets the backend service
        crypto_api_job = next(
            config for config in scrape_configs if config.get("job_name") == "crypto_api")
        static_configs = crypto_api_job.get("static_configs", [])
        targets = static_configs[0].get("targets", []) if static_configs else []
        self.assertIn("backend:8000", targets, "Prometheus not configured to target backend:8000")

    def test_grafana_datasource_provisioning(self):
        """Test that Grafana is provisioned with Prometheus datasource."""
        # Check if datasource.yml exists
        self.assertTrue(os.path.exists("grafana/provisioning/datasources/datasource.yml"),
                        "Grafana datasource provisioning file doesn't exist")

        # Read datasource.yml
        with open("grafana/provisioning/datasources/datasource.yml", "r") as f:
            datasource_config = yaml.safe_load(f)

        # Check for Prometheus datasource
        datasources = datasource_config.get("datasources", [])
        datasource_names = [ds.get("name") for ds in datasources]
        self.assertIn(
            "Prometheus",
            datasource_names,
            "Prometheus datasource not configured in Grafana")

        # Check it targets the Prometheus service
        prometheus_ds = next(ds for ds in datasources if ds.get("name") == "Prometheus")
        self.assertEqual(prometheus_ds.get("url"), "http://prometheus:9090",
                         "Prometheus datasource URL incorrect")

    def test_grafana_dashboard_provisioning(self):
        """Test that Grafana is provisioned with dashboards."""
        # Check if dashboards.yml exists
        self.assertTrue(os.path.exists("grafana/provisioning/dashboards/dashboards.yml"),
                        "Grafana dashboard provisioning file doesn't exist")

        # Check if fastapi_dashboard.json exists
        self.assertTrue(os.path.exists("grafana/dashboards/fastapi_dashboard.json"),
                        "Grafana FastAPI dashboard file doesn't exist")


if __name__ == "__main__":
    unittest.main()
