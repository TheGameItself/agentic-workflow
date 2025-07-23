"""
Context Manager Interface for the MCP system.

This module defines the interface for context management components.

λcontext_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid


class IContextManager(ABC):
    """Interface for context management components."""
    
    @abstractmethod
    def create_context(self, name: str, initial_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new context."""
        pass
    
    @abstractmethod
    def get_context(self, context_id: str) -> Dict[str, Any]:
        """Get a context by ID."""
        pass    
    
    @abstractmethod
    def update_context(self, context_id: str, data: Dict[str, Any], merge: bool = True) -> bool:
        """Update a context with new data."""
        pass
    
    @abstractmethod
    def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        pass
    
    @abstractmethod
    def list_contexts(self) -> List[Dict[str, Any]]:
        """List all contexts."""
        pass
    
    @abstractmethod
    def export_context(self, context_id: str, format_type: str = "json") -> str:
        """Export a context to a specified format."""
        pass
    
    @abstractmethod
    def import_context(self, context_data: str, format_type: str = "json") -> str:
        """Import a context from a specified format."""
        pass
    
    @abstractmethod
    def merge_contexts(self, context_ids: List[str], strategy: str = "override") -> str:
        """Merge multiple contexts into a new context."""
        pass
    
    @abstractmethod
    def get_context_history(self, context_id: str) -> List[Dict[str, Any]]:
        """Get the history of changes to a context."""
        pass