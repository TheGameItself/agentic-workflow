"""
Workflow Lobe Interface for the MCP system.

This module defines the interface for workflow lobe components.

Δworkflow_lobe_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from core.src.mcp.interfaces.lobe import ILobe


class IWorkflowLobe(ILobe):
    """Interface for workflow lobe components."""
    
    @abstractmethod
    def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow from a definition."""
        pass
    
    @abstractmethod
    def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow with optional input data."""
        pass
    
    @abstractmethod
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        pass
    
    @abstractmethod
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows managed by this lobe."""
        pass
    
    @abstractmethod
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow by ID."""
        pass