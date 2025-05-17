#!/usr/bin/env python3
"""
Script to automatically fix flake8 style issues in the codebase.
"""
import os
import subprocess
import sys
from pathlib import Path

def check_tools():
    """Verify required tools are installed."""
    required_tools = ['autopep8', 'isort', 'autoflake']
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run([tool, '--version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"Missing required tools: {', '.join(missing_tools)}")
        print("Installing missing tools...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_tools)

def fix_style_issues(directories):
    """Fix style issues in the specified directories."""
    print("Fixing style issues...")
    
    # Use autoflake to remove unused imports
    for directory in directories:
        print(f"Running autoflake on {directory}")
        subprocess.run([
            'autoflake',
            '--in-place',
            '--recursive',
            '--remove-all-unused-imports',
            '--remove-unused-variables',
            directory
        ])
    
    # Use isort to sort and clean up imports
    for directory in directories:
        print(f"Running isort on {directory}")
        subprocess.run(['isort', '--profile', 'black', directory])
    
    # Use autopep8 to fix most PEP8 issues
    for directory in directories:
        print(f"Running autopep8 on {directory}")
        subprocess.run([
            'autopep8',
            '--in-place',
            '--recursive',
            '--aggressive',
            '--aggressive',  # Use aggressive twice for more fixes
            '--max-line-length=100',
            directory
        ])

def run_flake8(directories):
    """Run flake8 to check if all issues are fixed."""
    print("\nRunning flake8 to check remaining issues...")
    result = subprocess.run(
        ['flake8'] + directories,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode == 0:
        print("All style issues fixed!")
        return True
    else:
        print("Some issues remain:")
        print(result.stdout)
        return False

def main():
    """Main function to run the style fixer."""
    directories = ['src/', 'tests/']
    
    # Make sure required tools are installed
    check_tools()
    
    # Fix style issues
    fix_style_issues(directories)
    
    # Run flake8 to check remaining issues
    run_flake8(directories)
    
    print("\nDone. Some issues may need manual fixing.")

if __name__ == "__main__":
    main() 