"""
Workflow Lobe Implementation for the MCP system.

This module provides an implementation of the IWorkflowLobe interface.

Δworkflow_lobe(implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

from core.src.mcp.interfaces.workflow_lobe import IWorkflowLobe
from core.src.mcp.lobes.base_lobe import BaseLobe
from core.src.mcp.implementations.workflow_engine import BasicWorkflowEngine
from core.src.mcp.exceptions import MCPLobeError, MCPWorkflowError

logger = logging.getLogger(__name__)


class WorkflowLobe(BaseLobe, IWorkflowLobe):
    """Implementation of the IWorkflowLobe interface."""
    
    def __init__(self, lobe_id: Optional[str] = None, name: Optional[str] = None, storage_path: Optional[str] = None):
        """Initialize the workflow lobe."""
        super().__init__(lobe_id, name or "WorkflowLobe")
        self.workflow_engine = BasicWorkflowEngine(storage_path)
        logger.info(f"WorkflowLobe {self.name} initialized with storage path: {storage_path}")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the lobe with optional configuration."""
        try:
            result = super().initialize(config)
            if not result:
                return False
            
            logger.info(f"WorkflowLobe {self.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowLobe {self.name}: {str(e)}")
            self.status = "error"
            return False
    
    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        try:
            if isinstance(input_data, dict):
                action = input_data.get("action")
                
                if action == "create_workflow":
                    return self.create_workflow(input_data.get("workflow_definition", {}))
                elif action == "execute_workflow":
                    return self.execute_workflow(
                        input_data.get("workflow_id"),
                        input_data.get("input_data")
                    )
                elif action == "get_status":
                    return self.get_workflow_status(input_data.get("workflow_id"))
                elif action == "list_workflows":
                    return self.list_workflows()
                elif action == "delete_workflow":
                    return self.delete_workflow(input_data.get("workflow_id"))
                else:
                    logger.warning(f"Unknown action in WorkflowLobe {self.name}: {action}")
                    return None
            else:
                logger.warning(f"Invalid input data format in WorkflowLobe {self.name}")
                return None
        except Exception as e:
            logger.error(f"Error processing data in WorkflowLobe {self.name}: {str(e)}")
            return None
    
    def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow from a definition."""
        try:
            # Extract workflow properties
            name = workflow_definition.get("name", "Unnamed Workflow")
            description = workflow_definition.get("description", "")
            
            # Create the workflow
            workflow_id = self.workflow_engine.create_workflow(name, description)
            
            # Add tasks if provided
            tasks = workflow_definition.get("tasks", [])
            for task_def in tasks:
                task_name = task_def.get("name", "Unnamed Task")
                task_function = task_def.get("function")
                dependencies = task_def.get("dependencies", [])
                
                if task_function:
                    self.workflow_engine.add_task(workflow_id, task_name, task_function, dependencies)
            
            logger.info(f"WorkflowLobe {self.name} created workflow {workflow_id} with {len(tasks)} tasks")
            return workflow_id
        except Exception as e:
            logger.error(f"Failed to create workflow in WorkflowLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to create workflow: {str(e)}")
    
    def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow with optional input data."""
        try:
            execution_result = self.workflow_engine.execute_workflow(workflow_id, input_data)
            logger.info(f"WorkflowLobe {self.name} executed workflow {workflow_id}")
            return execution_result
        except MCPWorkflowError as e:
            logger.error(f"Failed to execute workflow {workflow_id} in WorkflowLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to execute workflow: {str(e)}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        try:
            status = self.workflow_engine.get_workflow_status(workflow_id)
            logger.info(f"WorkflowLobe {self.name} retrieved status for workflow {workflow_id}")
            return status
        except MCPWorkflowError as e:
            logger.error(f"Failed to get workflow status {workflow_id} in WorkflowLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to get workflow status: {str(e)}")
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows managed by this lobe."""
        try:
            workflows = self.workflow_engine.list_workflows()
            logger.info(f"WorkflowLobe {self.name} listed {len(workflows)} workflows")
            return workflows
        except Exception as e:
            logger.error(f"Failed to list workflows in WorkflowLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to list workflows: {str(e)}")
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow by ID."""
        try:
            result = self.workflow_engine.delete_workflow(workflow_id)
            if result:
                logger.info(f"WorkflowLobe {self.name} deleted workflow {workflow_id}")
            else:
                logger.warning(f"WorkflowLobe {self.name} could not delete workflow {workflow_id}")
            return result
        except MCPWorkflowError as e:
            logger.error(f"Failed to delete workflow {workflow_id} in WorkflowLobe {self.name}: {str(e)}")
            raise MCPLobeError(f"Failed to delete workflow: {str(e)}")
    
    def receive_message(self, source_lobe_id: str, message: Any) -> bool:
        """Receive a message from another lobe."""
        try:
            logger.info(f"WorkflowLobe {self.name} received message from {source_lobe_id}")
            
            # Process the message as input data
            result = self.process(message)
            
            # Send response back if needed
            if result is not None and source_lobe_id in self.connections:
                response_message = {
                    "type": "response",
                    "original_message": message,
                    "result": result
                }
                return self.send_message(source_lobe_id, response_message)
            
            return True
        except Exception as e:
            logger.error(f"Failed to receive message in WorkflowLobe {self.name}: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the lobe."""
        base_status = super().get_status()
        try:
            workflows = self.workflow_engine.list_workflows()
            base_status.update({
                "total_workflows": len(workflows),
                "workflow_statuses": {w["id"]: w["status"] for w in workflows}
            })
        except Exception as e:
            logger.error(f"Failed to get workflow statistics in WorkflowLobe {self.name}: {str(e)}")
            base_status.update({
                "total_workflows": "unknown",
                "workflow_statuses": {}
            })
        
        return base_status