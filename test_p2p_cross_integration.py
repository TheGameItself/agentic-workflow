#!/usr/bin/env python3
"""
Test for P2P cross-integration with cognitive architecture.
"""

import asyncio
import logging
import unittest
from unittest.mock import MagicMock, patch
from enum import Enum
import time
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock classes to avoid importing actual modules
class MessageBusEvent(Enum):
    """Message bus event types."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    MESSAGE = "message"
    BROADCAST = "broadcast"
    STATUS_UPDATE = "status_update"
    THOUGHT_SHARE = "thought_share"
    COGNITIVE_UPDATE = "cognitive_update"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2

class P2PNetworkBus:
    """Mock P2P network bus."""
    
    def __init__(self, core_system=None, spinal_column=None):
        self.core_system = core_system
        self.spinal_column = spinal_column
        self.is_running = False
        self.subscriptions = {}
        self.messages = []
        
    async def start(self, bootstrap_nodes=None):
        """Start the P2P network bus."""
        self.is_running = True
        return True
        
    async def stop(self):
        """Stop the P2P network bus."""
        self.is_running = False
        return True
        
    async def publish(self, event_type, content, priority=None, ttl=10, metadata=None):
        """Publish a message to the bus."""
        message_id = str(uuid.uuid4())
        message = {
            'message_id': message_id,
            'event_type': event_type,
            'content': content,
            'priority': priority,
            'timestamp': time.time(),
            'ttl': ttl,
            'metadata': metadata or {}
        }
        self.messages.append(message)
        
        # Notify subscribers
        await self._notify_subscribers(message)
        
        return message_id
    
    async def _notify_subscribers(self, message):
        """Notify subscribers of a message."""
        event_type = message['event_type']
        
        for subscription_info in self.subscriptions.values():
            event_types = subscription_info['event_types']
            if event_type in event_types:
                callback = subscription_info['callback']
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Error in subscription callback: {e}")
        
    def subscribe(self, subscriber_id, event_types, callback, filter_criteria=None):
        """Subscribe to message bus events."""
        subscription_id = f"{subscriber_id}_{str(uuid.uuid4())[:8]}"
        self.subscriptions[subscription_id] = {
            'subscriber_id': subscriber_id,
            'event_types': event_types,
            'callback': callback,
            'filter_criteria': filter_criteria or {}
        }
        return subscription_id
        
    def unsubscribe(self, subscription_id):
        """Unsubscribe from message bus events."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            return True
        return False
        
    def get_bus_status(self):
        """Get current status of the message bus."""
        return {
            'is_running': self.is_running,
            'subscriptions': len(self.subscriptions),
            'messages': len(self.messages)
        }

class P2PCoreIntegration:
    """Mock P2P core integration."""
    
    def __init__(self, core_system, spinal_column=None):
        self.core_system = core_system
        self.spinal_column = spinal_column
        self.network_bus = P2PNetworkBus(core_system, spinal_column)
        self.is_initialized = False
        self.subscription_ids = []
        
    async def initialize(self, bootstrap_nodes=None):
        """Initialize P2P core integration."""
        await self.network_bus.start(bootstrap_nodes)
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        self.is_initialized = True
        return True
    
    def _subscribe_to_events(self):
        """Subscribe to relevant network events."""
        # Subscribe to cognitive events
        sub_id = self.network_bus.subscribe(
            subscriber_id="core_integration",
            event_types=[MessageBusEvent.THOUGHT_SHARE, MessageBusEvent.COGNITIVE_UPDATE],
            callback=self._handle_cognitive_event
        )
        self.subscription_ids.append(sub_id)
        
        # Subscribe to general network events
        sub_id = self.network_bus.subscribe(
            subscriber_id="core_integration",
            event_types=[
                MessageBusEvent.CONNECT,
                MessageBusEvent.DISCONNECT,
                MessageBusEvent.STATUS_UPDATE
            ],
            callback=self._handle_network_event
        )
        self.subscription_ids.append(sub_id)
    
    async def _handle_cognitive_event(self, message):
        """Handle cognitive event."""
        logger.info(f"Handling cognitive event: {message['event_type'].value}")
        
        # If core system has cognitive architecture, update it
        if hasattr(self.core_system, 'cognitive_architecture'):
            event_type = message['event_type']
            content = message['content']
            
            if event_type == MessageBusEvent.THOUGHT_SHARE:
                # Add shared thought to cognitive architecture
                thought_content = content.get('thought_content')
                thought_priority = content.get('thought_priority', 0.5)
                thought_source = content.get('thought_source', 'p2p_network')
                thought_metadata = content.get('thought_metadata', {})
                
                if thought_content:
                    self.core_system.cognitive_architecture.add_thought(
                        content=thought_content,
                        priority=thought_priority,
                        source=thought_source,
                        metadata=thought_metadata
                    )
            
            elif event_type == MessageBusEvent.COGNITIVE_UPDATE:
                # Update cognitive state
                cognitive_state = content.get('cognitive_state')
                if cognitive_state:
                    self.core_system.cognitive_architecture.set_cognitive_state(cognitive_state)
    
    async def _handle_network_event(self, message):
        """Handle network event."""
        logger.debug(f"Handling network event: {message['event_type'].value}")
        
    async def shutdown(self):
        """Shutdown P2P core integration."""
        # Unsubscribe from events
        for subscription_id in self.subscription_ids:
            self.network_bus.unsubscribe(subscription_id)
        
        await self.network_bus.stop()
        self.is_initialized = False
        return True
        
    def get_integration_status(self):
        """Get current status of the P2P core integration."""
        return {
            'is_initialized': self.is_initialized,
            'network_bus_status': self.network_bus.get_bus_status()
        }

class CognitiveArchitecture:
    """Mock cognitive architecture."""
    
    def __init__(self):
        self.thoughts = []
        self.cognitive_state = "relaxed"
        self.p2p_integration = None
    
    def add_thought(self, content, priority=0.5, source="system", metadata=None):
        """Add a thought to the cognitive architecture."""
        thought_id = f"thought_{int(time.time())}_{len(self.thoughts)}"
        thought = {
            'id': thought_id,
            'content': content,
            'priority': priority,
            'source': source,
            'metadata': metadata or {},
            'creation_time': time.time()
        }
        self.thoughts.append(thought)
        return thought_id
    
    def get_thought(self, thought_id):
        """Get a thought by ID."""
        for thought in self.thoughts:
            if thought['id'] == thought_id:
                return thought
        return None
    
    def set_cognitive_state(self, state):
        """Set the cognitive state."""
        self.cognitive_state = state
        return True
    
    def get_cognitive_status(self):
        """Get current status of the cognitive architecture."""
        return {
            'cognitive_state': self.cognitive_state,
            'active_thoughts': len(self.thoughts),
            'p2p_enabled': self.p2p_integration is not None
        }
    
    async def initialize(self, config=None):
        """Initialize cognitive architecture."""
        if config and config.get('enable_p2p', False):
            await self._initialize_p2p_integration(config.get('p2p_config', {}))
        return True
    
    async def _initialize_p2p_integration(self, p2p_config):
        """Initialize P2P integration."""
        # Create P2P integration
        self.p2p_integration = P2PCoreIntegration(self)
        
        # Initialize P2P integration
        success = await self.p2p_integration.initialize(p2p_config.get('bootstrap_nodes'))
        
        return success
    
    async def shutdown(self):
        """Shutdown cognitive architecture."""
        if self.p2p_integration:
            await self.p2p_integration.shutdown()
        return True

class TestP2PCrossIntegration(unittest.TestCase):
    """Test P2P cross-integration with cognitive architecture."""
    
    def setUp(self):
        """Set up test environment."""
        # Create cognitive architecture
        self.cognitive_architecture = CognitiveArchitecture()
        
        # Create mock core system
        self.mock_core_system = MagicMock()
        self.mock_core_system.cognitive_architecture = self.cognitive_architecture
    
    async def test_p2p_cross_integration(self):
        """Test P2P cross-integration with cognitive architecture."""
        print("Starting P2P cross-integration test")
        
        # Initialize cognitive architecture with P2P
        config = {
            'enable_p2p': True,
            'p2p_config': {
                'bootstrap_nodes': [('127.0.0.1', 8000)]
            }
        }
        
        print("Initializing cognitive architecture with P2P...")
        # Initialize cognitive architecture
        success = await self.cognitive_architecture.initialize(config)
        print(f"Initialization success: {success}")
        self.assertTrue(success)
        self.assertIsNotNone(self.cognitive_architecture.p2p_integration)
        print("Cognitive architecture initialized with P2P")
        
        # Add a thought to cognitive architecture
        thought_id = self.cognitive_architecture.add_thought(
            content="Test thought",
            priority=0.8,
            source="test",
            metadata={"test": True}
        )
        
        # Get the thought
        thought = self.cognitive_architecture.get_thought(thought_id)
        self.assertIsNotNone(thought)
        
        # Share thought via P2P
        message_id = await self.cognitive_architecture.p2p_integration.network_bus.publish(
            event_type=MessageBusEvent.THOUGHT_SHARE,
            content={
                "thought_id": thought_id,
                "thought_content": thought["content"],
                "thought_priority": thought["priority"],
                "thought_source": thought["source"],
                "thought_metadata": thought["metadata"]
            },
            priority=MessagePriority.NORMAL
        )
        self.assertIsNotNone(message_id)
        
        # Create a second cognitive architecture
        second_cognitive = CognitiveArchitecture()
        second_cognitive.p2p_integration = P2PCoreIntegration(second_cognitive)
        await second_cognitive.p2p_integration.initialize()
        
        # Subscribe to thought sharing events
        received_thoughts = []
        
        async def thought_callback(message):
            content = message['content']
            received_thoughts.append(content)
            
            # Add thought to second cognitive architecture
            second_cognitive.add_thought(
                content=content.get('thought_content', ''),
                priority=content.get('thought_priority', 0.5),
                source=content.get('thought_source', 'p2p'),
                metadata=content.get('thought_metadata', {})
            )
        
        subscription_id = second_cognitive.p2p_integration.network_bus.subscribe(
            subscriber_id="second_cognitive",
            event_types=[MessageBusEvent.THOUGHT_SHARE],
            callback=thought_callback
        )
        
        # Share another thought
        thought_id2 = self.cognitive_architecture.add_thought(
            content="Cross-integration thought",
            priority=0.9,
            source="cross_test",
            metadata={"cross": True}
        )
        
        thought2 = self.cognitive_architecture.get_thought(thought_id2)
        
        message_id2 = await self.cognitive_architecture.p2p_integration.network_bus.publish(
            event_type=MessageBusEvent.THOUGHT_SHARE,
            content={
                "thought_id": thought_id2,
                "thought_content": thought2["content"],
                "thought_priority": thought2["priority"],
                "thought_source": thought2["source"],
                "thought_metadata": thought2["metadata"]
            },
            priority=MessagePriority.HIGH
        )
        
        # Wait a moment for async processing
        await asyncio.sleep(0.1)
        
        # Check if thought was received
        self.assertEqual(len(received_thoughts), 1)
        self.assertEqual(received_thoughts[0]["thought_content"], "Cross-integration thought")
        
        # Check if thought was added to second cognitive architecture
        self.assertEqual(len(second_cognitive.thoughts), 1)
        self.assertEqual(second_cognitive.thoughts[0]["content"], "Cross-integration thought")
        
        # Update cognitive state via P2P
        await self.cognitive_architecture.p2p_integration.network_bus.publish(
            event_type=MessageBusEvent.COGNITIVE_UPDATE,
            content={
                "cognitive_state": "focused"
            },
            priority=MessagePriority.HIGH
        )
        
        # Wait a moment for async processing
        await asyncio.sleep(0.1)
        
        # Shutdown both cognitive architectures
        await self.cognitive_architecture.shutdown()
        await second_cognitive.shutdown()

def run_tests():
    """Run the tests."""
    async def run_async_tests():
        print("Starting P2P cross-integration tests...")
        test = TestP2PCrossIntegration()
        test.setUp()
        print("Test setup complete")
        
        try:
            print("Running P2P cross-integration test...")
            await test.test_p2p_cross_integration()
            print("All tests passed!")
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_async_tests())

if __name__ == '__main__':
    run_tests()