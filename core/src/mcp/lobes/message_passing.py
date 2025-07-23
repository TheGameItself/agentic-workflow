"""
Message Passing System for Lobe Communication in the MCP system.

This module provides a message passing interface that uses the event system.

Δmessage_passing(communication_implementation)
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import logging
from datetime import datetime
import threading
import time

from .event_system import EventBus, Event, EventType, EventPriority, EventHandler, get_event_bus

logger = logging.getLogger(__name__)


class MessageType:
    """Constants for message types."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


class Message:
    """Represents a message between lobes."""
    
    def __init__(self, message_type: str, sender_id: str, recipient_id: Optional[str] = None,
                 content: Any = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a message."""
        self.id = str(uuid.uuid4())
        self.message_type = message_type
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.correlation_id = None  # For request-response correlation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "message_type": self.message_type,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        message = cls(
            data["message_type"],
            data["sender_id"],
            data.get("recipient_id"),
            data.get("content"),
            data.get("metadata", {})
        )
        message.id = data["id"]
        message.timestamp = datetime.fromisoformat(data["timestamp"])
        message.correlation_id = data.get("correlation_id")
        return message


class MessageRouter:
    """Routes messages between lobes using the event system."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize the message router."""
        self.event_bus = event_bus or get_event_bus()
        self.registered_lobes = {}  # Dict[str, Dict] - lobe_id -> lobe_info
        self.message_handlers = {}  # Dict[str, Callable] - lobe_id -> handler
        self.pending_requests = {}  # Dict[str, Dict] - request_id -> request_info
        self.lock = threading.RLock()
        
        # Register event handler for message events
        self.event_handler = EventHandler(
            "message_router",
            [EventType.MESSAGE, EventType.REQUEST, EventType.RESPONSE],
            self._handle_event
        )
        self.event_bus.subscribe(self.event_handler)
        
        logger.info("MessageRouter initialized")
    
    def register_lobe(self, lobe_id: str, lobe_name: str, message_handler: Callable) -> bool:
        """Register a lobe with the message router."""
        try:
            with self.lock:
                self.registered_lobes[lobe_id] = {
                    "id": lobe_id,
                    "name": lobe_name,
                    "registered_at": datetime.now(),
                    "message_count": 0
                }
                self.message_handlers[lobe_id] = message_handler
            
            logger.info(f"Registered lobe {lobe_name} ({lobe_id}) with message router")
            return True
        except Exception as e:
            logger.error(f"Failed to register lobe {lobe_id}: {str(e)}")
            return False
    
    def unregister_lobe(self, lobe_id: str) -> bool:
        """Unregister a lobe from the message router."""
        try:
            with self.lock:
                if lobe_id in self.registered_lobes:
                    lobe_info = self.registered_lobes[lobe_id]
                    del self.registered_lobes[lobe_id]
                    del self.message_handlers[lobe_id]
                    logger.info(f"Unregistered lobe {lobe_info['name']} ({lobe_id})")
                    return True
                else:
                    logger.warning(f"Lobe {lobe_id} not found for unregistration")
                    return False
        except Exception as e:
            logger.error(f"Failed to unregister lobe {lobe_id}: {str(e)}")
            return False
    
    def send_message(self, message: Message) -> bool:
        """Send a message through the event system."""
        try:
            # Determine event type based on message type
            if message.message_type == MessageType.REQUEST:
                event_type = EventType.REQUEST
                priority = EventPriority.NORMAL
            elif message.message_type == MessageType.RESPONSE:
                event_type = EventType.RESPONSE
                priority = EventPriority.NORMAL
            elif message.message_type == MessageType.NOTIFICATION:
                event_type = EventType.NOTIFICATION
                priority = EventPriority.LOW
            else:
                event_type = EventType.MESSAGE
                priority = EventPriority.NORMAL
            
            # Create event
            event = Event(
                event_type,
                message.sender_id,
                message.recipient_id,
                message.to_dict(),
                priority
            )
            
            # For requests, track them for response correlation
            if message.message_type == MessageType.REQUEST:
                with self.lock:
                    self.pending_requests[message.id] = {
                        "message": message,
                        "sent_at": datetime.now(),
                        "timeout": 30.0  # 30 second timeout
                    }
            
            # Publish event
            result = self.event_bus.publish(event)
            
            if result:
                logger.debug(f"Sent message {message.id} from {message.sender_id} to {message.recipient_id}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {str(e)}")
            return False
    
    def send_direct_message(self, sender_id: str, recipient_id: str, content: Any, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a direct message to a specific lobe."""
        message = Message(MessageType.DIRECT, sender_id, recipient_id, content, metadata)
        return self.send_message(message)
    
    def send_broadcast(self, sender_id: str, content: Any, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a broadcast message to all registered lobes."""
        message = Message(MessageType.BROADCAST, sender_id, None, content, metadata)
        return self.send_message(message)
    
    def send_request(self, sender_id: str, recipient_id: str, content: Any,
                    metadata: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Optional[Any]:
        """Send a request and wait for response."""
        try:
            # Create request message
            request = Message(MessageType.REQUEST, sender_id, recipient_id, content, metadata)
            
            # Send the request
            if not self.send_message(request):
                return None
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self.lock:
                    if request.id not in self.pending_requests:
                        # Request was completed
                        break
                
                time.sleep(0.1)  # Brief pause
            
            # Check if we got a response
            with self.lock:
                if request.id in self.pending_requests:
                    # Timeout occurred
                    del self.pending_requests[request.id]
                    logger.warning(f"Request {request.id} timed out")
                    return None
            
            # Response should be available (handled by _handle_response)
            return None  # Response is handled asynchronously
        except Exception as e:
            logger.error(f"Failed to send request: {str(e)}")
            return None
    
    def send_notification(self, sender_id: str, recipient_id: Optional[str], content: Any,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a notification message."""
        message = Message(MessageType.NOTIFICATION, sender_id, recipient_id, content, metadata)
        return self.send_message(message)
    
    def _handle_event(self, event: Event) -> Any:
        """Handle events from the event system."""
        try:
            if event.data is None:
                return None
            
            # Convert event data back to message
            message = Message.from_dict(event.data)
            
            # Handle different event types
            if event.event_type == EventType.REQUEST:
                return self._handle_request(message)
            elif event.event_type == EventType.RESPONSE:
                return self._handle_response(message)
            elif event.event_type == EventType.MESSAGE or event.event_type == EventType.NOTIFICATION:
                return self._handle_message(message)
            
            return None
        except Exception as e:
            logger.error(f"Failed to handle event {event.id}: {str(e)}")
            return None
    
    def _handle_request(self, message: Message) -> Any:
        """Handle a request message."""
        try:
            # Find the target lobe
            if message.recipient_id and message.recipient_id in self.message_handlers:
                handler = self.message_handlers[message.recipient_id]
                
                # Call the handler
                response_content = handler(message)
                
                # Send response back
                if response_content is not None:
                    response = Message(
                        MessageType.RESPONSE,
                        message.recipient_id,
                        message.sender_id,
                        response_content
                    )
                    response.correlation_id = message.id
                    self.send_message(response)
                
                return response_content
            else:
                logger.warning(f"No handler found for request to {message.recipient_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to handle request {message.id}: {str(e)}")
            return None
    
    def _handle_response(self, message: Message) -> None:
        """Handle a response message."""
        try:
            # Find the original request
            correlation_id = message.correlation_id
            if correlation_id:
                with self.lock:
                    if correlation_id in self.pending_requests:
                        # Remove from pending requests
                        del self.pending_requests[correlation_id]
                        logger.debug(f"Received response for request {correlation_id}")
            
            # Also deliver to the recipient if they have a handler
            if message.recipient_id and message.recipient_id in self.message_handlers:
                handler = self.message_handlers[message.recipient_id]
                handler(message)
        except Exception as e:
            logger.error(f"Failed to handle response {message.id}: {str(e)}")
    
    def _handle_message(self, message: Message) -> None:
        """Handle a regular message or notification."""
        try:
            # Update message count
            with self.lock:
                if message.sender_id in self.registered_lobes:
                    self.registered_lobes[message.sender_id]["message_count"] += 1
            
            # Handle broadcast messages
            if message.message_type == MessageType.BROADCAST:
                # Send to all registered lobes except sender
                for lobe_id, handler in self.message_handlers.items():
                    if lobe_id != message.sender_id:
                        try:
                            handler(message)
                        except Exception as e:
                            logger.error(f"Failed to deliver broadcast to {lobe_id}: {str(e)}")
            else:
                # Send to specific recipient
                if message.recipient_id and message.recipient_id in self.message_handlers:
                    handler = self.message_handlers[message.recipient_id]
                    handler(message)
                else:
                    logger.warning(f"No handler found for message to {message.recipient_id}")
        except Exception as e:
            logger.error(f"Failed to handle message {message.id}: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message router statistics."""
        with self.lock:
            return {
                "registered_lobes": len(self.registered_lobes),
                "pending_requests": len(self.pending_requests),
                "lobe_info": dict(self.registered_lobes)
            }
    
    def cleanup_expired_requests(self) -> None:
        """Clean up expired pending requests."""
        try:
            current_time = datetime.now()
            expired_requests = []
            
            with self.lock:
                for request_id, request_info in self.pending_requests.items():
                    sent_at = request_info["sent_at"]
                    timeout = request_info["timeout"]
                    
                    if (current_time - sent_at).total_seconds() > timeout:
                        expired_requests.append(request_id)
                
                # Remove expired requests
                for request_id in expired_requests:
                    del self.pending_requests[request_id]
                    logger.warning(f"Cleaned up expired request {request_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup expired requests: {str(e)}")


# Global message router instance
_global_message_router = None
_router_lock = threading.Lock()


def get_message_router() -> MessageRouter:
    """Get the global message router instance."""
    global _global_message_router
    
    with _router_lock:
        if _global_message_router is None:
            _global_message_router = MessageRouter()
    
    return _global_message_router