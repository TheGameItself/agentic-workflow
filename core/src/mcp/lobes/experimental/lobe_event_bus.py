"""
LobeEventBus: Event bus for inter-lobe communication.
"""

import logging
from typing import Any, Dict, List, Callable, Optional


class LobeEventBus:
    """
    Event bus for inter-lobe communication in the MCP system.
    Allows lobes to communicate through events without direct dependencies.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}
        self.logger = logging.getLogger("LobeEventBus")
        
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
            
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to event type: {event_type}")
        
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed from event type: {event_type}")
            
    def emit(self, event_type: str, data: Any = None) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event_type: Type of event to emit
            data: Data to pass to subscribers
        """
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
                    
        self.logger.debug(f"Emitted event type: {event_type}")
        
    def predictive_broadcast(self, event_type: str, data: Any = None, context: Dict = None) -> None:
        """
        Predictively broadcast an event with context information.
        
        Args:
            event_type: Type of event to emit
            data: Data to pass to subscribers
            context: Additional context information
        """
        # Add context to data if provided
        if context and isinstance(data, dict):
            if isinstance(data, dict):
                data["_context"] = context
            else:
                data = {"value": data, "_context": context}
                
        # Emit event
        self.emit(event_type, data)
        
        # Predictively emit related events
        if event_type == "brain_state_update":
            self.emit("hormone_levels_updated", data.get("hormone_levels", {}))
            
        self.logger.debug(f"Predictively broadcast event type: {event_type}")