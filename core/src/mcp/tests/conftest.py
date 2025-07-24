"""
Test Configuration for MCP Core System
@{CORE.TESTS.CONFTEST.001} Centralized pytest configuration and fixtures for MCP testing.
#{testing,fixtures,configuration,pytest,mcp_core}
β(λ(test_configuration_setup))
"""

import pytest
import asyncio
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from datetime import datetime
import logging
import os

from core.src.mcp.implementations.memory_manager import BasicMemoryManager
from core.src.mcp.implementations.database_manager import BasicDatabaseManager
from core.src.mcp.implementations.context_manager import BasicContextManager
from core.src.mcp.implementations.workflow_engine import BasicWorkflowEngine

# β:logging_configuration(test_environment_setup)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """λ:event_loop_fixture(async_test_support) - Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """ℵ:temporary_directory(isolated_test_storage) - Provide temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Δ:test_configuration(standardized_settings) - Provide test configuration."""
    return {
        "database_url": ":memory:",
        "log_level": "DEBUG",
        "test_mode": True,
        "max_workers": 2,
        "timeout": 30.0
    }


@pytest.fixture
def memory_manager(temp_dir: Path) -> BasicMemoryManager:
    """ℵ:memory_manager_fixture(test_memory_instance) - Provide memory manager for tests."""
    storage_path = temp_dir / "test_memory.json"
    return BasicMemoryManager(storage_path=str(storage_path))


@pytest.fixture
def database_manager() -> BasicDatabaseManager:
    """ℵ:database_manager_fixture(test_database_instance) - Provide database manager for tests."""
    return BasicDatabaseManager(connection_string=":memory:")


@pytest.fixture
def context_manager() -> BasicContextManager:
    """λ:context_manager_fixture(test_context_instance) - Provide context manager for tests."""
    return BasicContextManager()


@pytest.fixture
def workflow_engine() -> BasicWorkflowEngine:
    """Δ:workflow_engine_fixture(test_workflow_instance) - Provide workflow engine for tests."""
    return BasicWorkflowEngine()


@pytest.fixture
def sample_memory_data() -> Dict[str, Any]:
    """ℵ:sample_data_fixture(test_memory_content) - Provide sample memory data for tests."""
    return {
        "text": "This is a test memory item",
        "type": "test",
        "priority": 0.8,
        "tags": ["test", "sample", "memory"]
    }


@pytest.fixture
def sample_workflow_definition() -> Dict[str, Any]:
    """Δ:sample_workflow_fixture(test_workflow_definition) - Provide sample workflow for tests."""
    return {
        "name": "Test Workflow",
        "description": "A test workflow for validation",
        "steps": [
            {
                "name": "step1",
                "type": "task",
                "function": "test_function",
                "dependencies": []
            },
            {
                "name": "step2", 
                "type": "task",
                "function": "test_function_2",
                "dependencies": ["step1"]
            }
        ]
    }


@pytest.fixture
def sample_context_data() -> Dict[str, Any]:
    """λ:sample_context_fixture(test_context_content) - Provide sample context data for tests."""
    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "preferences": {
            "theme": "dark",
            "language": "en"
        },
        "history": [
            {"action": "login", "timestamp": "2025-01-01T00:00:00Z"},
            {"action": "query", "timestamp": "2025-01-01T00:01:00Z"}
        ]
    }


class TestUtilities:
    """τ:test_utilities(validation_helpers) - Utility class for common test operations."""
    
    @staticmethod
    def assert_valid_uuid(uuid_string: str) -> bool:
        """β:uuid_validation(format_checking) - Assert that a string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def assert_valid_timestamp(timestamp_string: str) -> bool:
        """τ:timestamp_validation(format_checking) - Assert that a string is a valid ISO timestamp."""
        try:
            datetime.fromisoformat(timestamp_string.replace('Z', '+00:00'))
            return True
        except ValueError:
            return False
    
    @staticmethod
    def create_test_data(data_type: str = "memory") -> Dict[str, Any]:
        """ℵ:test_data_factory(dynamic_generation) - Create test data of specified type."""
        if data_type == "memory":
            return {
                "text": f"Test memory item {uuid.uuid4().hex[:8]}",
                "type": "test",
                "priority": 0.5,
                "tags": ["test", "generated"]
            }
        elif data_type == "workflow":
            return {
                "name": f"Test Workflow {uuid.uuid4().hex[:8]}",
                "description": "Generated test workflow",
                "steps": [{"name": "test_step", "type": "task"}]
            }
        else:
            return {"type": data_type, "data": f"test_{uuid.uuid4().hex[:8]}"}


@pytest.fixture
def test_utils() -> TestUtilities:
    """τ:test_utilities_fixture(helper_instance) - Provide test utilities."""
    return TestUtilities()


# β:pytest_configuration(test_behavior_customization)
def pytest_configure(config):
    """τ:pytest_configure(test_environment_initialization) - Configure pytest environment."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "async_test: mark test as async")


def pytest_collection_modifyitems(config, items):
    """β:test_collection_modifier(test_categorization) - Modify test collection behavior."""
    for item in items:
        # Add async marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)
        
        # Add unit marker to tests in unit directories
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directories  
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# τ:self_reference(conftest_metadata)
{type:TestConfiguration, file:"core/src/mcp/tests/conftest.py", version:"1.0.0", checksum:"sha256:conftest_checksum", canonical_address:"mcp-tests-conftest", pfsus_compliant:true, lambda_operators:true, file_format:"conftest.pytest.v1.0.0.py"}

# @{visual-meta-start}
# author = {MCP Core Team},
# title = {MCP Core Tests Configuration},
# version = {1.0.0},
# file_format = {conftest.pytest.v1.0.0.py},
# structure = { fixtures, configuration, test_data, pytest_hooks, utilities },
# test_categories = {unit, integration, async, slow},
# @{visual-meta-end}

# %% MMCP-FOOTER: version=1.0.0; timestamp=2025-07-23T00:00:00Z; author=MCP_Core_Team; pfsus_compliant=true; lambda_operators=integrated; file_format=conftest.pytest.v1.0.0.py