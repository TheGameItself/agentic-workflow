"""
Integration tests for the MCP system.

This module tests the interaction between different components.

βintegration_tests(component_interaction)
"""

import pytest
import uuid
from typing import Any, Dict

from core.src.mcp.implementations.memory_manager import BasicMemoryManager
from core.src.mcp.implementations.workflow_engine import BasicWorkflowEngine
from core.src.mcp.implementations.context_manager import BasicContextManager
from core.src.mcp.implementations.database_manager import BasicDatabaseManager
from core.src.mcp.lobes.memory_lobe import MemoryLobe
from core.src.mcp.lobes.workflow_lobe import WorkflowLobe
from core.src.mcp.lobes.context_lobe import ContextLobe


class TestComponentIntegration:
    """Test integration between core components."""
    
    def test_memory_workflow_integration(self, memory_manager: BasicMemoryManager, 
                                       workflow_engine: BasicWorkflowEngine,
                                       sample_data: Dict[str, Any],
                                       sample_workflow: Dict[str, Any]):
        """Test integration between memory manager and workflow engine."""
        # Store data in memory
        data_key = memory_manager.store(sample_data)
        assert data_key is not None
        
        # Create workflow that references the stored data
        workflow_data = sample_workflow.copy()
        workflow_data["steps"][0]["parameters"]["data_key"] = data_key
        
        workflow_id = workflow_engine.create_workflow(workflow_data)
        assert workflow_id is not None
        
        # Verify workflow can access stored data
        workflow = workflow_engine.get_workflow(workflow_id)
        assert workflow is not None
        assert workflow["steps"][0]["parameters"]["data_key"] == data_key
        
        # Verify data is still accessible
        retrieved_data = memory_manager.retrieve(data_key)
        assert retrieved_data == sample_data
    
    def test_context_memory_integration(self, context_manager: BasicContextManager,
                                      memory_manager: BasicMemoryManager,
                                      sample_context: Dict[str, Any],
                                      sample_data: Dict[str, Any]):
        """Test integration between context manager and memory manager."""
        # Store data in memory
        data_key = memory_manager.store(sample_data)
        
        # Create context that references the stored data
        context_data = sample_context.copy()
        context_data["memory_references"] = [data_key]
        
        context_id = context_manager.create_context(context_data)
        assert context_id is not None
        
        # Verify context can reference stored data
        context = context_manager.get_context(context_id)
        assert context is not None
        assert data_key in context["memory_references"]
        
        # Verify data is accessible through context
        retrieved_data = memory_manager.retrieve(data_key)
        assert retrieved_data == sample_data
    
    def test_database_memory_integration(self, database_manager: BasicDatabaseManager,
                                       memory_manager: BasicMemoryManager,
                                       sample_data: Dict[str, Any]):
        """Test integration between database manager and memory manager."""
        # Store data in memory
        data_key = memory_manager.store(sample_data)
        
        # Store reference in database
        query = "INSERT INTO memory_references (key, metadata) VALUES (?, ?)"
        params = (data_key, str(sample_data))
        
        # Create table first
        create_table_query = """
        CREATE TABLE IF NOT EXISTS memory_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        database_manager.execute_query(create_table_query)
        database_manager.execute_query(query, params)
        
        # Verify reference exists in database
        select_query = "SELECT key, metadata FROM memory_references WHERE key = ?"
        result = database_manager.execute_query(select_query, (data_key,))
        assert len(result) == 1
        assert result[0][0] == data_key
        
        # Verify data is still accessible in memory
        retrieved_data = memory_manager.retrieve(data_key)
        assert retrieved_data == sample_data


class TestLobeIntegration:
    """Test integration between different lobes."""
    
    def test_memory_workflow_lobe_integration(self, temp_dir: str,
                                            sample_data: Dict[str, Any],
                                            sample_workflow: Dict[str, Any]):
        """Test integration between memory and workflow lobes."""
        # Create lobes
        memory_lobe = MemoryLobe()
        workflow_lobe = WorkflowLobe()
        
        # Initialize lobes
        memory_config = {"storage_path": f"{temp_dir}/memory_lobe.json"}
        workflow_config = {"storage_path": f"{temp_dir}/workflow_lobe.json"}
        
        assert memory_lobe.initialize(memory_config)
        assert workflow_lobe.initialize(workflow_config)
        
        # Connect lobes
        assert memory_lobe.connect(workflow_lobe, "workflow_integration")
        assert workflow_lobe.connect(memory_lobe, "memory_integration")
        
        # Test data flow between lobes
        # Memory lobe stores data
        store_result = memory_lobe.process({
            "action": "store",
            "data": sample_data
        })
        assert "key" in store_result
        data_key = store_result["key"]
        
        # Workflow lobe creates workflow referencing the data
        workflow_data = sample_workflow.copy()
        workflow_data["memory_key"] = data_key
        
        workflow_result = workflow_lobe.process({
            "action": "create_workflow",
            "workflow": workflow_data
        })
        assert "workflow_id" in workflow_result
        
        # Verify integration
        connections = memory_lobe.get_connections()
        assert len(connections) == 1
        assert connections[0]["type"] == "workflow_integration"
        
        # Cleanup
        memory_lobe.shutdown()
        workflow_lobe.shutdown()
    
    def test_context_memory_lobe_integration(self, temp_dir: str,
                                           sample_context: Dict[str, Any],
                                           sample_data: Dict[str, Any]):
        """Test integration between context and memory lobes."""
        # Create lobes
        context_lobe = ContextLobe()
        memory_lobe = MemoryLobe()
        
        # Initialize lobes
        context_config = {"storage_path": f"{temp_dir}/context_lobe.json"}
        memory_config = {"storage_path": f"{temp_dir}/memory_lobe.json"}
        
        assert context_lobe.initialize(context_config)
        assert memory_lobe.initialize(memory_config)
        
        # Connect lobes
        assert context_lobe.connect(memory_lobe, "memory_integration")
        assert memory_lobe.connect(context_lobe, "context_integration")
        
        # Test data flow
        # Store data in memory lobe
        store_result = memory_lobe.process({
            "action": "store",
            "data": sample_data
        })
        data_key = store_result["key"]
        
        # Create context that references the data
        context_data = sample_context.copy()
        context_data["memory_references"] = [data_key]
        
        context_result = context_lobe.process({
            "action": "create_context",
            "context": context_data
        })
        assert "context_id" in context_result
        
        # Verify integration
        connections = context_lobe.get_connections()
        assert len(connections) == 1
        assert connections[0]["type"] == "memory_integration"
        
        # Cleanup
        context_lobe.shutdown()
        memory_lobe.shutdown()


class TestSystemIntegration:
    """Test end-to-end system integration."""
    
    def test_full_system_workflow(self, temp_dir: str,
                                 sample_data: Dict[str, Any],
                                 sample_workflow: Dict[str, Any],
                                 sample_context: Dict[str, Any]):
        """Test a complete workflow through the entire system."""
        # Create all components
        memory_lobe = MemoryLobe()
        workflow_lobe = WorkflowLobe()
        context_lobe = ContextLobe()
        
        # Initialize all components
        memory_config = {"storage_path": f"{temp_dir}/memory.json"}
        workflow_config = {"storage_path": f"{temp_dir}/workflow.json"}
        context_config = {"storage_path": f"{temp_dir}/context.json"}
        
        assert memory_lobe.initialize(memory_config)
        assert workflow_lobe.initialize(workflow_config)
        assert context_lobe.initialize(context_config)
        
        # Connect all components
        assert memory_lobe.connect(workflow_lobe, "workflow")
        assert memory_lobe.connect(context_lobe, "context")
        assert workflow_lobe.connect(memory_lobe, "memory")
        assert workflow_lobe.connect(context_lobe, "context")
        assert context_lobe.connect(memory_lobe, "memory")
        assert context_lobe.connect(workflow_lobe, "workflow")
        
        # Execute full workflow
        # 1. Store initial data
        store_result = memory_lobe.process({
            "action": "store",
            "data": sample_data
        })
        data_key = store_result["key"]
        
        # 2. Create context
        context_data = sample_context.copy()
        context_data["data_key"] = data_key
        
        context_result = context_lobe.process({
            "action": "create_context",
            "context": context_data
        })
        context_id = context_result["context_id"]
        
        # 3. Create workflow
        workflow_data = sample_workflow.copy()
        workflow_data["context_id"] = context_id
        workflow_data["data_key"] = data_key
        
        workflow_result = workflow_lobe.process({
            "action": "create_workflow",
            "workflow": workflow_data
        })
        workflow_id = workflow_result["workflow_id"]
        
        # 4. Verify all components are connected and working
        assert memory_lobe.get_status()["status"] == "ready"
        assert workflow_lobe.get_status()["status"] == "ready"
        assert context_lobe.get_status()["status"] == "ready"
        
        # 5. Verify data integrity across components
        retrieved_data = memory_lobe.process({
            "action": "retrieve",
            "key": data_key
        })
        assert retrieved_data["data"] == sample_data
        
        # Cleanup
        memory_lobe.shutdown()
        workflow_lobe.shutdown()
        context_lobe.shutdown()