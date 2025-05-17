import os
import unittest


class TestProjectStructure(unittest.TestCase):
    """Test the project structure is set up correctly."""

    def test_directories_exist(self):
        """Test that all required directories exist."""
        required_dirs = [
            'data/raw',
            'data/processed',
            'data/external',
            'data/interim',
            'notebooks',
            'src/data',
            'src/features',
            'src/models',
            'src/visualization',
            'src/config',
            'src/app',
            'models',
            'reports/figures',
            'tests',
            '.github/workflows',
        ]

        for directory in required_dirs:
            self.assertTrue(
                os.path.isdir(directory),
                f"Directory {directory} does not exist"
            )

    def test_config_files_exist(self):
        """Test that all required configuration files exist."""
        required_files = [
            'README.md',
            'requirements.txt',
            'dvc.yaml',
            'Dockerfile',
            'docker-compose.yml',
            '.flake8',
            '.env',
            '.gitignore',
        ]

        for file in required_files:
            self.assertTrue(
                os.path.isfile(file),
                f"File {file} does not exist"
            )


if __name__ == '__main__':
    unittest.main()
