"""
Memory Lobe Interface for the MCP system.

This module defines the interface for memory lobe components.

ℵmemory_lobe_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from core.src.mcp.interfaces.lobe import ILobe


class IMemoryLobe(ILobe):
    """Interface for memory lobe components."""
    
    @abstractmethod
    def store_memory(self, memory_data: Any, memory_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory of the specified type."""
        pass
    
    @abstractmethod
    def retrieve_memory(self, memory_id: str) -> Any:
        """Retrieve a memory by ID."""
        pass
    
    @abstractmethod
    def search_memories(self, query: str, memory_type: Optional[str] = None, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search memories by query and optional type."""
        pass
    
    @abstractmethod
    def forget_memory(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        pass
    
    @abstractmethod
    def get_memory_types(self) -> List[str]:
        """Get all available memory types."""
        pass