"""
Context Lobe Interface for the MCP system.

This module defines the interface for context lobe components.

λcontext_lobe_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from core.src.mcp.interfaces.lobe import ILobe


class IContextLobe(ILobe):
    """Interface for context lobe components."""
    
    @abstractmethod
    def create_context(self, context_type: str, initial_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new context of the specified type."""
        pass
    
    @abstractmethod
    def get_context(self, context_id: str) -> Dict[str, Any]:
        """Get a context by ID."""
        pass
    
    @abstractmethod
    def update_context(self, context_id: str, data: Dict[str, Any]) -> bool:
        """Update a context with new data."""
        pass
    
    @abstractmethod
    def delete_context(self, context_id: str) -> bool:
        """Delete a context by ID."""
        pass
    
    @abstractmethod
    def list_contexts(self, context_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all contexts, optionally filtered by type."""
        pass