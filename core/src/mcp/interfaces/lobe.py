"""
Base Lobe Interface for the MCP system.

This module defines the interface for all lobe components in the brain-inspired architecture.

Ωlobe_interface(standardized_definition)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid


class ILobe(ABC):
    """Interface for all lobe components."""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the lobe with optional configuration."""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        pass 
   
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the lobe."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the lobe gracefully."""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset the lobe to its initial state."""
        pass
    
    @abstractmethod
    def connect(self, other_lobe: 'ILobe', connection_type: str) -> bool:
        """Connect this lobe to another lobe."""
        pass
    
    @abstractmethod
    def disconnect(self, other_lobe: 'ILobe') -> bool:
        """Disconnect this lobe from another lobe."""
        pass
    
    @abstractmethod
    def send_message(self, target_lobe_id: str, message: Any) -> bool:
        """Send a message to another lobe."""
        pass
    
    @abstractmethod
    def receive_message(self, source_lobe_id: str, message: Any) -> bool:
        """Receive a message from another lobe."""
        pass
    
    @abstractmethod
    def get_connections(self) -> List[Dict[str, Any]]:
        """Get all connections to other lobes."""
        pass