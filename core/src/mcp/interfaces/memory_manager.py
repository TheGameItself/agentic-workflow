"""
Memory Manager Interface for the MCP system.

This module defines the interface for memory management components.

ℵmemory_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid


class IMemoryManager(ABC):
    """Interface for memory management components."""
    
    @abstractmethod
    def store(self, data: Any, key: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data in memory."""
        pass 
   
    @abstractmethod
    def retrieve(self, key: str) -> Any:
        """Retrieve data from memory."""
        pass
    
    @abstractmethod
    def update(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update data in memory."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data from memory."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for data in memory."""
        pass
    
    @abstractmethod
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a stored item."""
        pass
    
    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys in memory."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all data from memory."""
        pass