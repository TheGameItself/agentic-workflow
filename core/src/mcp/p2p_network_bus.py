#!/usr/bin/env python3
"""
P2P Network Bus for MCP Core System
Implements a message bus for P2P network communication and integration with core tools.
"""

import asyncio
import logging
import os
import time
import json
import threading
import random
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
import queue

# Import P2P network components
from .p2p_network import P2PNetworkNode, MessageType, NetworkNode
from .p2p_network_integration import P2PNetworkIntegration, UserStatus, ServerCapability, NetworkRegion

# Import core components
from .core_system import MCPCoreSystem
from .spinal_column import SpinalColumn, NeuralPathway

class MessageBusEvent(Enum):
    """Message bus event types."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    MESSAGE = "message"
    BROADCAST = "broadcast"
    DISCOVERY = "discovery"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    GENETIC_EXCHANGE = "genetic_exchange"
    ENGRAM_TRANSFER = "engram_transfer"
    RESEARCH_UPDATE = "research_update"
    MODEL_SYNC = "model_sync"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class MessageBusSubscription:
    """Subscription to message bus events."""
    
    def __init__(self, 
                 subscriber_id: str,
                 event_types: List[MessageBusEvent],
                 callback: Callable,
                 filter_criteria: Optional[Dict[str, Any]] = None):
        """Initialize subscription."""
        self.subscriber_id = subscriber_id
        self.event_types = event_types
        self.callback = callback
        self.filter_criteria = filter_criteria or {}
        self.subscription_time = time.time()
        self.last_invoked = None
        self.invocation_count = 0class Mes
sageBusMessage:
    """Message for the P2P network bus."""
    
    def __init__(self,
                 message_id: str,
                 event_type: MessageBusEvent,
                 sender_id: str,
                 content: Dict[str, Any],
                 priority: MessagePriority = MessagePriority.NORMAL,
                 timestamp: float = None,
                 ttl: int = 10,
                 metadata: Dict[str, Any] = None):
        """Initialize message."""
        self.message_id = message_id
        self.event_type = event_type
        self.sender_id = sender_id
        self.content = content
        self.priority = priority
        self.timestamp = timestamp or time.time()
        self.ttl = ttl
        self.metadata = metadata or {}
        self.processed_by: Set[str] = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'event_type': self.event_type.value,
            'sender_id': self.sender_id,
            'content': self.content,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageBusMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            event_type=MessageBusEvent(data['event_type']),
            sender_id=data['sender_id'],
            content=data['content'],
            priority=MessagePriority(data['priority']),
            timestamp=data['timestamp'],
            ttl=data['ttl'],
            metadata=data.get('metadata', {})
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.ttl <= 0
    
    def decrement_ttl(self) -> int:
        """Decrement TTL and return new value."""
        self.ttl -= 1
        return self.ttl
    
    def mark_processed_by(self, processor_id: str):
        """Mark message as processed by a specific processor."""
        self.processed_by.add(processor_id)
    
    def was_processed_by(self, processor_id: str) -> bool:
        """Check if message was processed by a specific processor."""
        return processor_id in self.processed_bycla
ss P2PNetworkBus:
    """
    P2P Network Bus for MCP Core System.
    
    Features:
    - Message routing between P2P network and core components
    - Event-based subscription system
    - Priority-based message processing
    - Asynchronous message handling
    - Integration with spinal column for neural processing
    - Support for genetic exchange and engram transfer
    - Research data synchronization
    - Model weight synchronization
    """
    
    def __init__(self, 
                 node_id: str = None,
                 port: int = None,
                 core_system: Optional[MCPCoreSystem] = None,
                 spinal_column: Optional[SpinalColumn] = None):
        """Initialize P2P network bus."""
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port or random.randint(10000, 65000)
        self.core_system = core_system
        self.spinal_column = spinal_column
        
        # Network components
        self.network_node = P2PNetworkNode(node_id=self.node_id, port=self.port)
        self.network_integration = P2PNetworkIntegration(enable_research_tracking=True)
        
        # Message bus components
        self.subscriptions: Dict[str, MessageBusSubscription] = {}
        self.message_queue = asyncio.PriorityQueue()
        self.message_history: List[MessageBusMessage] = []
        self.max_history_size = 1000
        
        # Processing state
        self.is_running = False
        self.processing_task = None
        self.message_processors: Dict[MessageBusEvent, Callable] = {}
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'events_by_type': {event.value: 0 for event in MessageBusEvent},
            'subscription_invocations': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        self.logger = logging.getLogger("p2p_network_bus")
        
        # Initialize message processors
        self._setup_message_processors()
    
    def _setup_message_processors(self):
        """Setup message processors for different event types."""
        self.message_processors = {
            MessageBusEvent.CONNECT: self._process_connect_event,
            MessageBusEvent.DISCONNECT: self._process_disconnect_event,
            MessageBusEvent.MESSAGE: self._process_message_event,
            MessageBusEvent.BROADCAST: self._process_broadcast_event,
            MessageBusEvent.DISCOVERY: self._process_discovery_event,
            MessageBusEvent.STATUS_UPDATE: self._process_status_update_event,
            MessageBusEvent.ERROR: self._process_error_event,
            MessageBusEvent.SHUTDOWN: self._process_shutdown_event,
            MessageBusEvent.GENETIC_EXCHANGE: self._process_genetic_exchange_event,
            MessageBusEvent.ENGRAM_TRANSFER: self._process_engram_transfer_event,
            MessageBusEvent.RESEARCH_UPDATE: self._process_research_update_event,
            MessageBusEvent.MODEL_SYNC: self._process_model_sync_event,
            MessageBusEvent.RESOURCE_REQUEST: self._process_resource_request_event,
            MessageBusEvent.RESOURCE_RESPONSE: self._process_resource_response_event
        }
    
    async def start(self, bootstrap_nodes: Optional[List[Tuple[str, int]]] = None):
        """Start the P2P network bus."""
        if self.is_running:
            self.logger.warning("P2P network bus already running")
            return
        
        self.logger.info(f"Starting P2P network bus (Node ID: {self.node_id})")
        
        # Start P2P network node
        await self.network_node.start(bootstrap_nodes)
        
        # Start network integration
        await self.network_integration.start()
        
        # Register self as user
        self.network_integration.register_user(
            user_id=self.node_id,
            username=f"Node_{self.node_id[:8]}",
            capability=ServerCapability.ADVANCED,
            region=NetworkRegion.LOCAL,
            expertise_domains=["neural_networks", "genetic_algorithms", "memory_systems"]
        )
        
        # Start message processing
        self.is_running = True
        self.processing_task = asyncio.create_task(self._message_processing_loop())
        
        self.logger.info("P2P network bus started successfully")
    
    async def stop(self):
        """Stop the P2P network bus."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping P2P network bus")
        
        # Stop message processing
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Stop network integration
        await self.network_integration.stop()
        
        # Stop P2P network node
        await self.network_node.stop()
        
        self.logger.info("P2P network bus stopped")
    
    async def _message_processing_loop(self):
        """Main message processing loop."""
        while self.is_running:
            try:
                # Get next message from queue
                priority, message = await self.message_queue.get()
                
                # Process message
                await self._process_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(0.1)
    
    async def _process_message(self, message: MessageBusMessage):
        """Process a message from the queue."""
        try:
            # Check if message has expired
            if message.is_expired():
                self.logger.debug(f"Skipping expired message: {message.message_id}")
                return
            
            # Decrement TTL
            message.decrement_ttl()
            
            # Get processor for event type
            processor = self.message_processors.get(message.event_type)
            if processor:
                await processor(message)
            else:
                self.logger.warning(f"No processor for event type: {message.event_type}")
            
            # Update statistics
            self.stats['messages_processed'] += 1
            self.stats['events_by_type'][message.event_type.value] += 1
            
            # Add to history
            self._add_to_history(message)
            
            # Notify subscribers
            await self._notify_subscribers(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {e}")
            self.stats['errors'] += 1
    
    def _add_to_history(self, message: MessageBusMessage):
        """Add message to history."""
        self.message_history.append(message)
        
        # Limit history size
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    async def _notify_subscribers(self, message: MessageBusMessage):
        """Notify subscribers of a message."""
        event_type = message.event_type
        
        for subscription in self.subscriptions.values():
            if event_type in subscription.event_types:
                # Check filter criteria
                if self._matches_filter(message, subscription.filter_criteria):
                    try:
                        # Invoke callback
                        if asyncio.iscoroutinefunction(subscription.callback):
                            await subscription.callback(message)
                        else:
                            subscription.callback(message)
                        
                        # Update subscription stats
                        subscription.last_invoked = time.time()
                        subscription.invocation_count += 1
                        self.stats['subscription_invocations'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error in subscription callback: {e}")
                        self.stats['errors'] += 1
    
    def _matches_filter(self, message: MessageBusMessage, filter_criteria: Dict[str, Any]) -> bool:
        """Check if message matches filter criteria."""
        if not filter_criteria:
            return True
        
        for key, value in filter_criteria.items():
            if key == 'sender_id' and message.sender_id != value:
                return False
            elif key == 'content' and not self._content_matches(message.content, value):
                return False
            elif key == 'priority' and message.priority != value:
                return False
            elif key == 'metadata' and not self._content_matches(message.metadata, value):
                return False
        
        return True
    
    def _content_matches(self, content: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if content matches criteria."""
        for key, value in criteria.items():
            if key not in content or content[key] != value:
                return False
        return True 
   def subscribe(self, 
                subscriber_id: str,
                event_types: List[MessageBusEvent],
                callback: Callable,
                filter_criteria: Optional[Dict[str, Any]] = None) -> str:
        """Subscribe to message bus events."""
        subscription_id = f"{subscriber_id}_{str(uuid.uuid4())[:8]}"
        
        subscription = MessageBusSubscription(
            subscriber_id=subscriber_id,
            event_types=event_types,
            callback=callback,
            filter_criteria=filter_criteria
        )
        
        self.subscriptions[subscription_id] = subscription
        self.logger.debug(f"Added subscription: {subscription_id} for {subscriber_id}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from message bus events."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            self.logger.debug(f"Removed subscription: {subscription_id}")
            return True
        
        return False
    
    async def publish(self, 
                    event_type: MessageBusEvent,
                    content: Dict[str, Any],
                    priority: MessagePriority = MessagePriority.NORMAL,
                    ttl: int = 10,
                    metadata: Dict[str, Any] = None) -> str:
        """Publish a message to the bus."""
        message_id = str(uuid.uuid4())
        
        message = MessageBusMessage(
            message_id=message_id,
            event_type=event_type,
            sender_id=self.node_id,
            content=content,
            priority=priority,
            timestamp=time.time(),
            ttl=ttl,
            metadata=metadata
        )
        
        # Add to queue with priority
        await self.message_queue.put((priority.value, message))
        
        # Update statistics
        self.stats['messages_sent'] += 1
        
        self.logger.debug(f"Published message: {message_id} ({event_type.value})")
        
        return message_id
    
    async def broadcast_to_network(self, 
                                 event_type: MessageBusEvent,
                                 content: Dict[str, Any],
                                 priority: MessagePriority = MessagePriority.NORMAL,
                                 ttl: int = 10,
                                 metadata: Dict[str, Any] = None) -> str:
        """Broadcast a message to the P2P network."""
        # Create message
        message_id = str(uuid.uuid4())
        
        message = MessageBusMessage(
            message_id=message_id,
            event_type=event_type,
            sender_id=self.node_id,
            content=content,
            priority=priority,
            timestamp=time.time(),
            ttl=ttl,
            metadata=metadata
        )
        
        # Convert to P2P network message
        p2p_message_type = self._map_event_to_message_type(event_type)
        p2p_message = {
            'message_id': message_id,
            'event_type': event_type.value,
            'content': content,
            'priority': priority.value,
            'timestamp': time.time(),
            'ttl': ttl,
            'metadata': metadata or {}
        }
        
        # Broadcast to network
        # TODO: Implement actual broadcast using P2P network node
        
        # Also process locally
        await self.message_queue.put((priority.value, message))
        
        # Update statistics
        self.stats['messages_sent'] += 1
        
        self.logger.debug(f"Broadcast message to network: {message_id} ({event_type.value})")
        
        return message_id
    
    def _map_event_to_message_type(self, event_type: MessageBusEvent) -> MessageType:
        """Map bus event type to P2P message type."""
        mapping = {
            MessageBusEvent.CONNECT: MessageType.HANDSHAKE,
            MessageBusEvent.DISCONNECT: MessageType.HEARTBEAT,
            MessageBusEvent.MESSAGE: MessageType.STORE,
            MessageBusEvent.BROADCAST: MessageType.FIND_VALUE,
            MessageBusEvent.DISCOVERY: MessageType.FIND_NODE,
            MessageBusEvent.STATUS_UPDATE: MessageType.NETWORK_STATUS,
            MessageBusEvent.GENETIC_EXCHANGE: MessageType.GENETIC_DATA,
            MessageBusEvent.ENGRAM_TRANSFER: MessageType.ENGRAM_DATA,
            MessageBusEvent.MODEL_SYNC: MessageType.STORE,
            MessageBusEvent.RESOURCE_REQUEST: MessageType.BANDWIDTH_TEST,
            MessageBusEvent.RESOURCE_RESPONSE: MessageType.BANDWIDTH_TEST
        }
        
        return mapping.get(event_type, MessageType.STORE)
    
    async def send_to_spinal_column(self, message: MessageBusMessage) -> bool:
        """Send message to spinal column for neural processing."""
        if not self.spinal_column:
            self.logger.warning("No spinal column available")
            return False
        
        try:
            # Convert message content to tensor
            import torch
            import json
            
            # Serialize message to JSON and convert to tensor
            message_json = json.dumps(message.to_dict())
            message_bytes = message_json.encode('utf-8')
            
            # Create tensor from bytes
            data = torch.tensor([float(b) for b in message_bytes], dtype=torch.float32)
            
            # Determine pathway type based on message priority
            pathway_mapping = {
                MessagePriority.LOW: NeuralPathway.ASCENDING,
                MessagePriority.NORMAL: NeuralPathway.LATERAL,
                MessagePriority.HIGH: NeuralPathway.DESCENDING,
                MessagePriority.CRITICAL: NeuralPathway.REFLEXIVE
            }
            
            pathway = pathway_mapping.get(message.priority, NeuralPathway.ASCENDING)
            
            # Process through spinal column
            result = self.spinal_column.process_signal(
                data=data,
                pathway_type=pathway,
                priority=message.priority.value,
                metadata={'message_id': message.message_id, 'event_type': message.event_type.value}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message to spinal column: {e}")
            return False
    
    # Event processors
    async def _process_connect_event(self, message: MessageBusMessage):
        """Process connect event."""
        self.logger.info(f"Processing connect event from {message.sender_id}")
        # Implementation details...
    
    async def _process_disconnect_event(self, message: MessageBusMessage):
        """Process disconnect event."""
        self.logger.info(f"Processing disconnect event from {message.sender_id}")
        # Implementation details...
    
    async def _process_message_event(self, message: MessageBusMessage):
        """Process message event."""
        self.logger.debug(f"Processing message event from {message.sender_id}")
        # Implementation details...
    
    async def _process_broadcast_event(self, message: MessageBusMessage):
        """Process broadcast event."""
        self.logger.debug(f"Processing broadcast event from {message.sender_id}")
        # Implementation details...
    
    async def _process_discovery_event(self, message: MessageBusMessage):
        """Process discovery event."""
        self.logger.debug(f"Processing discovery event from {message.sender_id}")
        # Implementation details...
    
    async def _process_status_update_event(self, message: MessageBusMessage):
        """Process status update event."""
        self.logger.debug(f"Processing status update event from {message.sender_id}")
        # Implementation details...
    
    async def _process_error_event(self, message: MessageBusMessage):
        """Process error event."""
        self.logger.warning(f"Processing error event from {message.sender_id}: {message.content.get('error')}")
        # Implementation details...
    
    async def _process_shutdown_event(self, message: MessageBusMessage):
        """Process shutdown event."""
        self.logger.info(f"Processing shutdown event from {message.sender_id}")
        # Implementation details...
    
    async def _process_genetic_exchange_event(self, message: MessageBusMessage):
        """Process genetic exchange event."""
        self.logger.info(f"Processing genetic exchange event from {message.sender_id}")
        # Implementation details...
    
    async def _process_engram_transfer_event(self, message: MessageBusMessage):
        """Process engram transfer event."""
        self.logger.info(f"Processing engram transfer event from {message.sender_id}")
        # Implementation details...
    
    async def _process_research_update_event(self, message: MessageBusMessage):
        """Process research update event."""
        self.logger.info(f"Processing research update event from {message.sender_id}")
        # Implementation details...
    
    async def _process_model_sync_event(self, message: MessageBusMessage):
        """Process model sync event."""
        self.logger.info(f"Processing model sync event from {message.sender_id}")
        # Implementation details...
    
    async def _process_resource_request_event(self, message: MessageBusMessage):
        """Process resource request event."""
        self.logger.debug(f"Processing resource request event from {message.sender_id}")
        # Implementation details...
    
    async def _process_resource_response_event(self, message: MessageBusMessage):
        """Process resource response event."""
        self.logger.debug(f"Processing resource response event from {message.sender_id}")
        # Implementation details...
    
    def get_bus_status(self) -> Dict[str, Any]:
        """Get current status of the message bus."""
        return {
            'node_id': self.node_id,
            'is_running': self.is_running,
            'queue_size': self.message_queue.qsize() if self.message_queue else 0,
            'subscriptions': len(self.subscriptions),
            'history_size': len(self.message_history),
            'stats': self.stats,
            'uptime': time.time() - self.stats['start_time']
        }

# Convenience function
def create_p2p_network_bus(core_system: Optional[MCPCoreSystem] = None,
                          spinal_column: Optional[SpinalColumn] = None) -> P2PNetworkBus:
    """Create a P2P network bus instance."""
    return P2PNetworkBus(core_system=core_system, spinal_column=spinal_column)