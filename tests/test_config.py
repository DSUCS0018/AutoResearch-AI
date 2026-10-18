"""
tests/test_config.py
--------------------
Minimal configuration tests for Stage 1.

These tests verify that the Config class correctly loads values
from the .env file and falls back to sensible defaults.
"""

import pytest
from app.config import config


def test_app_name():
    """Application name must match the expected value."""
    assert config.APP_NAME == "AutoResearch AI"


def test_app_version():
    """Application version must be set correctly."""
    assert config.APP_VERSION == "0.1.0"


def test_environment():
    """Environment should be 'development' for local runs."""
    assert config.ENVIRONMENT == "development"


def test_log_level_is_uppercase():
    """Log level must always be stored in uppercase."""
    assert config.LOG_LEVEL == config.LOG_LEVEL.upper()


def test_log_level_is_valid():
    """Log level must be one of the standard Python logging levels."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    assert config.LOG_LEVEL in valid_levels


def test_project_root_exists():
    """PROJECT_ROOT path must resolve to an existing directory."""
    assert config.PROJECT_ROOT.is_dir()
