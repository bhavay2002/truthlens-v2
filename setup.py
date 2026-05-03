"""
Setup script for TruthLens AI
Automates initial setup and checks
"""
import os
import sys
from pathlib import Path
import subprocess
import logging
from src.utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def check_python_version():
    """Ensure Python 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8+ required!")
        logger.error(f"Current version: {sys.version}")
        sys.exit(1)
    logger.info(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")


def create_directories():
    """Create necessary directories"""
    dirs = [
        Path("data/raw"),
        Path("data/processed"),
        Path("data/interim"),
        Path("models"),
        Path("logs"),
        Path("reports"),
        Path("experiments"),
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Directories created")


def check_dependencies():
    """Check if requirements are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import transformers
        import torch
        logger.info("✓ Core dependencies installed")
        return True
    except ImportError as e:
        logger.warning(f"Missing dependency: {e}")
        return False


def install_dependencies():
    """Install dependencies from requirements.txt"""
    logger.info("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        logger.error("Failed to install dependencies")
        return False


def check_data_files():
    """Check if data files exist"""
    fake_path = Path("data/raw/fake.csv")
    real_path = Path("data/raw/real.csv")
    
    if not fake_path.exists() or not real_path.exists():
        logger.warning("⚠ Data files not found!")
        logger.info("Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        logger.info(f"Place files at:")
        logger.info(f"  - {fake_path}")
        logger.info(f"  - {real_path}")
        return False
    else:
        logger.info("✓ Data files found")
        return True


def create_env_file():
    """Create .env from .env.example if it doesn't exist"""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            logger.info("✓ Created .env file from .env.example")
        else:
            logger.warning("⚠ .env.example not found")
    else:
        logger.info("✓ .env file exists")


def run_tests():
    """Run basic tests"""
    logger.info("Running tests...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ Tests passed")
            return True
        else:
            logger.warning("⚠ Some tests failed")
            return False
    except Exception as e:
        logger.warning(f"⚠ Could not run tests: {e}")
        return False


def main():
    """Run setup"""
    logger.info("=" * 50)
    logger.info("TruthLens AI - Setup Script")
    logger.info("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Check/install dependencies
    if not check_dependencies():
        logger.info("Installing missing dependencies...")
        if not install_dependencies():
            logger.error("Setup failed: Could not install dependencies")
            sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Check data files
    data_ok = check_data_files()
    
    # Run tests
    tests_ok = run_tests()
    
    # Summary
    logger.info("=" * 50)
    logger.info("Setup Summary:")
    logger.info(f"  Dependencies: ✓")
    logger.info(f"  Data files: {'✓' if data_ok else '⚠ Missing'}")
    logger.info(f"  Tests: {'✓' if tests_ok else '⚠ Check manually'}")
    logger.info("=" * 50)
    
    if data_ok:
        logger.info("\nNext steps:")
        logger.info("  1. Train model: python main.py")
        logger.info("  2. Run API: uvicorn api.app:app --reload")
        logger.info("  3. Visit: http://localhost:8000/docs")
    else:
        logger.info("\nNext steps:")
        logger.info("  1. Download data files (see instructions above)")
        logger.info("  2. Run setup again: python setup.py")
        logger.info("  3. Train model: python main.py")
    
    logger.info("\n✨ Setup complete!\n")


if __name__ == "__main__":
    main()
