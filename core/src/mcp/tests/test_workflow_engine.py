"""
Tests for the BasicWorkflowEngine implementation.

βtest_workflow_engine(functionality_validation)
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import time

from core.src.mcp.implementations.workflow_engine import BasicWorkflowEngine
from core.src.mcp.exceptions import MCPWorkflowError


class TestBasicWorkflowEngine(unittest.TestCase):
    """Test cases for the BasicWorkflowEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.workflow_engine = BasicWorkflowEngine()
    
    def test_create_workflow(self):
        """Test creating a workflow."""
        # Create workflow
        workflow_id = self.workflow_engine.create_workflow("Test Workflow", "Test description")
        
        # Verify workflow was created
        self.assertIsNotNone(workflow_id)
        
        # Get workflow
        workflow = self.workflow_engine.get_workflow(workflow_id)
        
        # Verify workflow properties
        self.assertEqual(workflow["name"], "Test Workflow")
        self.assertEqual(workflow["description"], "Test description")
        self.assertEqual(len(workflow["tasks"]), 0)
    
    def test_add_task(self):
        """Test adding a task to a workflow."""
        # Create workflow
        workflow_id = self.workflow_engine.create_workflow("Test Workflow")
        
        # Define a task function
        def task_function(data):
            return {"result": data.get("input", 0) + 1}
        
        # Add task
        task_id = self.workflow_engine.add_task(workflow_id, "Test Task", task_function)
        
        # Verify task was added
        self.assertIsNotNone(task_id)
        
        # Get workflow
        workflow = self.workflow_engine.get_workflow(workflow_id)
        
        # Verify task is in workflow
        self.assertEqual(len(workflow["tasks"]), 1)
        self.assertEqual(workflow["tasks"][0]["id"], task_id)
        self.assertEqual(workflow["tasks"][0]["name"], "Test Task")
    
    def test_execute_workflow(self):
        """Test executing a workflow."""
        # Create workflow
        workflow_id = self.workflow_engine.create_workflow("Test Workflow")
        
        # Define task functions
        def task1(data):
            return {"step1": data.get("input", 0) + 1}
        
        def task2(data):
            return {"step2": data.get("step1", 0) * 2}
        
        # Add tasks
        task1_id = self.workflow_engine.add_task(workflow_id, "Task 1", task1)
        task2_id = self.workflow_engine.add_task(workflow_id, "Task 2", task2, [task1_id])
        
        # Execute workflow
        execution = self.workflow_engine.execute_workflow(workflow_id, {"input": 5})
        
        # Verify execution started
        self.assertEqual(execution["status"], "running")
        
        # Wait for execution to complete
        execution_id = execution["execution_id"]
        max_wait = 5  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.workflow_engine.executions[execution_id]["status"] != "running":
                break
            time.sleep(0.1)
        
        # Verify execution completed
        self.assertEqual(self.workflow_engine.executions[execution_id]["status"], "completed")
        
        # Verify results
        output_data = self.workflow_engine.executions[execution_id]["output_data"]
        self.assertEqual(output_data["step1"], 6)  # 5 + 1
        self.assertEqual(output_data["step2"], 12)  # 6 * 2
    
    def test_list_workflows(self):
        """Test listing workflows."""
        # Create workflows
        workflow1_id = self.workflow_engine.create_workflow("Workflow 1")
        workflow2_id = self.workflow_engine.create_workflow("Workflow 2")
        
        # List workflows
        workflows = self.workflow_engine.list_workflows()
        
        # Verify workflows are listed
        self.assertEqual(len(workflows), 2)
        workflow_names = [w["name"] for w in workflows]
        self.assertIn("Workflow 1", workflow_names)
        self.assertIn("Workflow 2", workflow_names)
    
    def test_delete_workflow(self):
        """Test deleting a workflow."""
        # Create workflow
        workflow_id = self.workflow_engine.create_workflow("Test Workflow")
        
        # Delete workflow
        result = self.workflow_engine.delete_workflow(workflow_id)
        
        # Verify deletion was successful
        self.assertTrue(result)
        
        # Verify workflow no longer exists
        with self.assertRaises(MCPWorkflowError):
            self.workflow_engine.get_workflow(workflow_id)
    
    def test_workflow_with_dependencies(self):
        """Test workflow with task dependencies."""
        # Create workflow
        workflow_id = self.workflow_engine.create_workflow("Test Workflow")
        
        # Define task functions
        execution_order = []
        
        def task1(data):
            execution_order.append("task1")
            return {"step1": True}
        
        def task2(data):
            execution_order.append("task2")
            return {"step2": True}
        
        def task3(data):
            execution_order.append("task3")
            return {"step3": True}
        
        # Add tasks with dependencies
        task1_id = self.workflow_engine.add_task(workflow_id, "Task 1", task1)
        task2_id = self.workflow_engine.add_task(workflow_id, "Task 2", task2, [task1_id])
        task3_id = self.workflow_engine.add_task(workflow_id, "Task 3", task3, [task2_id])
        
        # Execute workflow
        execution = self.workflow_engine.execute_workflow(workflow_id)
        
        # Wait for execution to complete
        execution_id = execution["execution_id"]
        max_wait = 5  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.workflow_engine.executions[execution_id]["status"] != "running":
                break
            time.sleep(0.1)
        
        # Verify execution completed
        self.assertEqual(self.workflow_engine.executions[execution_id]["status"], "completed")
        
        # Verify execution order
        self.assertEqual(execution_order, ["task1", "task2", "task3"])


if __name__ == "__main__":
    unittest.main()