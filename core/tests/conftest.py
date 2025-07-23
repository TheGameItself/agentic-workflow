#!/usr/bin/env python3
"""
Pytest configuration file for MCP tests.
Provides common fixtures and configuration for all test files.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add core/src to path for testing
core_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(core_src))


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for the test session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def create_test_file():
    """Fixture to create a temporary test file."""
    temp_files = []
    
    def _create_file(content, suffix=".txt"):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w") as f:
            f.write(content)
            temp_files.append(f.name)
        return f.name
    
    yield _create_file
    
    # Cleanup
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.unlink(file_path)


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return {
        "name": "Test Project",
        "version": "1.0.0",
        "description": "A test project for MCP",
        "settings": {
            "debug": True,
            "log_level": "INFO",
            "max_connections": 5
        },
        "components": [
            {"name": "component1", "enabled": True},
            {"name": "component2", "enabled": False},
            {"name": "component3", "enabled": True}
        ]
    }