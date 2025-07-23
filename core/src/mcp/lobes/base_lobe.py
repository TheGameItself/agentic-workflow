"""
Base Lobe Implementation for the MCP system.

This module provides a base implementation of the ILobe interface.

Ωbase_lobe(implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

from core.src.mcp.interfaces.lobe import ILobe
from core.src.mcp.exceptions import MCPLobeError
from .event_system import get_event_system, Event, EventType, EventPriority

logger = logging.getLogger(__name__)


class BaseLobe(ILobe):
    """Base implementation of the ILobe interface."""
    
    def __init__(self, lobe_id: Optional[str] = None, name: Optional[str] = None):
        """Initialize the base lobe."""
        self.lobe_id = lobe_id or str(uuid.uuid4())
        self.name = name or f"{self.__class__.__name__}-{self.lobe_id[:8]}"
        self.connections = {}  # Dictionary of connected lobes
        self.status = "initialized"
        self.config = {}
        self.message_router = get_message_router()
        logger.info(f"Lobe {self.name} ({self.lobe_id}) initialized")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the lobe with optional configuration."""
        try:
            self.config = config or {}
            
            # Register with message router
            self.message_router.register_lobe(self.lobe_id, self.name, self._handle_message)
            
            self.status = "ready"
            logger.info(f"Lobe {self.name} initialized with config: {self.config}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize lobe {self.name}: {str(e)}")
            self.status = "error"
            return False    

    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        logger.warning(f"Base process method called on {self.name}. This should be overridden.")
        return input_data
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the lobe."""
        return {
            "lobe_id": self.lobe_id,
            "name": self.name,
            "status": self.status,
            "connections": len(self.connections),
            "config": self.config
        }
    
    def shutdown(self) -> bool:
        """Shutdown the lobe gracefully."""
        try:
            # Unregister from message router
            self.message_router.unregister_lobe(self.lobe_id)
            
            # Disconnect from all connected lobes
            for lobe_id in list(self.connections.keys()):
                self.disconnect(self.connections[lobe_id]["lobe"])
            
            self.status = "shutdown"
            logger.info(f"Lobe {self.name} shutdown successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown lobe {self.name}: {str(e)}")
            return False
    
    def reset(self) -> bool:
        """Reset the lobe to its initial state."""
        try:
            # Keep connections but reset internal state
            old_connections = self.connections
            self.__init__(lobe_id=self.lobe_id, name=self.name)
            self.connections = old_connections
            logger.info(f"Lobe {self.name} reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset lobe {self.name}: {str(e)}")
            return False
    
    def connect(self, other_lobe: 'ILobe', connection_type: str) -> bool:
        """Connect this lobe to another lobe."""
        try:
            other_status = other_lobe.get_status()
            other_id = other_status["lobe_id"]
            
            if other_id in self.connections:
                logger.warning(f"Lobe {self.name} already connected to {other_status['name']}")
                return True
            
            self.connections[other_id] = {
                "lobe": other_lobe,
                "type": connection_type,
                "name": other_status["name"]
            }
            
            logger.info(f"Lobe {self.name} connected to {other_status['name']} ({connection_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect lobe {self.name}: {str(e)}")
            return False
    
    def disconnect(self, other_lobe: 'ILobe') -> bool:
        """Disconnect this lobe from another lobe."""
        try:
            other_status = other_lobe.get_status()
            other_id = other_status["lobe_id"]
            
            if other_id not in self.connections:
                logger.warning(f"Lobe {self.name} not connected to {other_status['name']}")
                return True
            
            del self.connections[other_id]
            logger.info(f"Lobe {self.name} disconnected from {other_status['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect lobe {self.name}: {str(e)}")
            return False
    
    def send_message(self, target_lobe_id: str, message: Any) -> bool:
        """Send a message to another lobe."""
        try:
            return self.message_router.send_direct_message(self.lobe_id, target_lobe_id, message)
        except Exception as e:
            logger.error(f"Failed to send message from lobe {self.name}: {str(e)}")
            return False
    
    def receive_message(self, source_lobe_id: str, message: Any) -> bool:
        """Receive a message from another lobe."""
        logger.warning(f"Base receive_message method called on {self.name}. This should be overridden.")
        return True
    
    def send_broadcast(self, message: Any) -> bool:
        """Send a broadcast message to all connected lobes."""
        try:
            return self.message_router.send_broadcast(self.lobe_id, message)
        except Exception as e:
            logger.error(f"Failed to send broadcast from lobe {self.name}: {str(e)}")
            return False
    
    def send_request(self, target_lobe_id: str, request: Any, timeout: float = 30.0) -> Optional[Any]:
        """Send a request to another lobe and wait for response."""
        try:
            return self.message_router.send_request(self.lobe_id, target_lobe_id, request, timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to send request from lobe {self.name}: {str(e)}")
            return None
    
    def send_notification(self, target_lobe_id: Optional[str], notification: Any) -> bool:
        """Send a notification to a specific lobe or broadcast."""
        try:
            return self.message_router.send_notification(self.lobe_id, target_lobe_id, notification)
        except Exception as e:
            logger.error(f"Failed to send notification from lobe {self.name}: {str(e)}")
            return False
    
    def _handle_message(self, message: Message) -> Any:
        """Handle incoming messages from the message router."""
        try:
            # Convert message to the old format for backward compatibility
            if message.message_type == MessageType.REQUEST:
                # Handle request and return response
                response = self.process(message.content)
                return response
            else:
                # Handle regular message/notification
                self.receive_message(message.sender_id, message.content)
                return None
        except Exception as e:
            logger.error(f"Failed to handle message in lobe {self.name}: {str(e)}")
            return None
    
    def get_connections(self) -> List[Dict[str, Any]]:
        """Get all connections to other lobes."""
        return [
            {
                "lobe_id": lobe_id,
                "name": info["name"],
                "type": info["type"]
            }
            for lobe_id, info in self.connections.items()
        ]