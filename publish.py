#!/usr/bin/env python3
"""
Script to build and publish the inferno-llm package to PyPI.
"""

import os
import subprocess
import sys
import shutil
import argparse


def check_dependencies():
    """Ensure required publishing dependencies are installed."""
    try:
        import setuptools
        import wheel
        import twine
    except ImportError:
        print("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", 
                              "setuptools", "wheel", "twine"])
        print("Dependencies installed successfully.")


def clean_build_dirs():
    """Remove build artifacts from previous builds."""
    dirs_to_clean = ["build", "dist", "inferno_llm.egg-info"]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)
    
    print("Build directories cleaned.")


def build_package():
    """Build source distribution and wheel."""
    print("Building distribution packages...")
    try:
        subprocess.check_call([sys.executable, "setup.py", "sdist", "bdist_wheel"])
        print("Package build successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error building package: {e}")
        sys.exit(1)


def upload_to_pypi(test=False):
    """Upload the package to PyPI or TestPyPI."""
    try:
        if test:
            print("Uploading to TestPyPI...")
            cmd = [sys.executable, "-m", "twine", "upload", "--repository-url", 
                  "https://test.pypi.org/legacy/", "dist/*"]
        else:
            print("Uploading to PyPI...")
            cmd = [sys.executable, "-m", "twine", "upload", "dist/*"]
        
        # This will prompt for PyPI username and password
        subprocess.check_call(cmd)
        
        if test:
            print("\nSuccessfully uploaded to TestPyPI!")
            print("You can view your package at: https://test.pypi.org/project/inferno-llm/")
            print("\nTo install from TestPyPI:")
            print("pip install --index-url https://test.pypi.org/simple/ inferno-llm")
        else:
            print("\nSuccessfully uploaded to PyPI!")
            print("You can view your package at: https://pypi.org/project/inferno-llm/")
            print("\nTo install from PyPI:")
            print("pip install inferno-llm")
            
    except subprocess.CalledProcessError as e:
        print(f"Error uploading to {'TestPyPI' if test else 'PyPI'}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build and publish inferno-llm to PyPI")
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Upload to TestPyPI instead of production PyPI"
    )
    args = parser.parse_args()
    
    # Make sure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Starting publication process for inferno-llm...")
    
    # Ensure we have the necessary dependencies
    check_dependencies()
    
    # Clean previous build artifacts
    clean_build_dirs()
    
    # Build the package
    build_package()
    
    # Upload to PyPI
    upload_to_pypi(test=args.test)
    clean_build_dirs()
    print("\nPublication process completed.")


if __name__ == "__main__":
    main()