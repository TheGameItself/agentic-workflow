"""
Workflow Engine Interface for the MCP system.

This module defines the interface for workflow management components.

Δworkflow_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid


class IWorkflowEngine(ABC):
    """Interface for workflow engine components."""
    
    @abstractmethod
    def create_workflow(self, name: str, description: Optional[str] = None) -> str:
        """Create a new workflow."""
        pass
    
    @abstractmethod
    def add_task(self, workflow_id: str, task_name: str, task_function: callable, 
                dependencies: Optional[List[str]] = None) -> str:
        """Add a task to a workflow."""
        pass  
  
    @abstractmethod
    def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow with optional input data."""
        pass
    
    @abstractmethod
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a workflow."""
        pass
    
    @abstractmethod
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        pass
    
    @abstractmethod
    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get details of a specific workflow."""
        pass
    
    @abstractmethod
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        pass
    
    @abstractmethod
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        pass
    
    @abstractmethod
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        pass