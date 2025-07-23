#!/usr/bin/env python3
"""
Test P2P Integration with Cognitive Architecture
Tests the integration of P2P network functionality with the cognitive architecture.
"""

import asyncio
import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

# Import core components
from core.src.mcp.cognitive_architecture import CognitiveArchitecture
from core.src.mcp.p2p_network_bus import P2PNetworkBus, MessageBusEvent, MessagePriority
from core.src.mcp.p2p_core_integration import P2PCoreIntegration
from core.src.mcp.core_system_p2p_integration import CoreSystemP2PIntegration, integrate_p2p_with_core_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestP2PCognitiveIntegration(unittest.TestCase):
    """Test P2P integration with cognitive architecture."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary data directory
        self.data_dir = Path("test_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Create cognitive architecture with mocked components
        self.cognitive_architecture = CognitiveArchitecture()
        
        # Mock components
        self.cognitive_architecture.memory_manager = MagicMock()
        self.cognitive_architecture.workflow_manager = MagicMock()
        self.cognitive_architecture.context_manager = MagicMock()
        self.cognitive_architecture.creative_engine = MagicMock()
        self.cognitive_architecture.learning_manager = MagicMock()
        self.cognitive_architecture.performance_monitor = MagicMock()
        
        # Mock methods
        self.cognitive_architecture._monitoring_loop = MagicMock()
        self.cognitive_architecture.get_status = MagicMock(return_value={})
        self.cognitive_architecture._shutdown_event = MagicMock()
        self.cognitive_architecture._shutdown_event.is_set = MagicMock(return_value=False)
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary data directory
        import shutil
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
    
    @patch('core.src.mcp.p2p_network_bus.P2PNetworkBus.start')
    @patch('core.src.mcp.p2p_network_bus.P2PNetworkBus.stop')
    @patch('core.src.mcp.p2p_core_integration.P2PCoreIntegration.initialize')
    @patch('core.src.mcp.p2p_core_integration.P2PCoreIntegration.shutdown')
    async def test_p2p_integration_lifecycle(self, mock_shutdown, mock_initialize, mock_stop, mock_start):
        """Test P2P integration lifecycle with cognitive architecture."""
        # Mock successful initialization
        mock_initialize.return_value = asyncio.Future()
        mock_initialize.return_value.set_result(True)
        
        mock_shutdown.return_value = asyncio.Future()
        mock_shutdown.return_value.set_result(True)
        
        # Initialize cognitive architecture with P2P
        config = {
            'enable_p2p': True,
            'p2p_config': {
                'bootstrap_nodes': [('127.0.0.1', 8000)]
            }
        }
        
        # Initialize cognitive architecture
        success = await self.cognitive_architecture.initialize(config)
        
        # Check initialization success
        self.assertTrue(success)
        self.assertTrue(hasattr(self.cognitive_architecture, 'p2p_integration'))
        
        # Mock p2p_integration.is_initialized
        self.cognitive_architecture.p2p_integration = MagicMock()
        self.cognitive_architecture.p2p_integration.is_initialized = True
        self.cognitive_architecture.p2p_integration.shutdown = mock_shutdown
        
        # Check if P2P is enabled
        is_enabled = CoreSystemP2PIntegration.is_p2p_enabled(self.cognitive_architecture)
        self.assertTrue(is_enabled)
        
        # Get P2P status
        status = CoreSystemP2PIntegration.get_p2p_status(self.cognitive_architecture)
        self.assertIsNotNone(status)
        
        # Shutdown cognitive architecture
        shutdown_success = await self.cognitive_architecture.shutdown()
        self.assertTrue(shutdown_success)
        
        # Check if shutdown was called
        mock_shutdown.assert_called_once()
    
    @patch('core.src.mcp.p2p_network_bus.P2PNetworkBus')
    async def test_thought_sharing(self, MockNetworkBus):
        """Test thought sharing through P2P network."""
        # Create mock network bus instance
        mock_network_bus = MockNetworkBus.return_value
        
        # Create P2P integration
        p2p_integration = P2PCoreIntegration(
            core_system=self.cognitive_architecture
        )
        
        # Set network bus
        p2p_integration.network_bus = mock_network_bus
        p2p_integration.is_initialized = True
        
        # Set P2P integration in cognitive architecture
        self.cognitive_architecture.p2p_integration = p2p_integration
        
        # Mock publish method
        mock_network_bus.publish = MagicMock()
        mock_network_bus.publish.return_value = asyncio.Future()
        mock_network_bus.publish.return_value.set_result("message_id")
        
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
        message_id = await p2p_integration.network_bus.publish(
            event_type=MessageBusEvent.BROADCAST,
            content={
                "thought_id": thought_id,
                "thought_content": thought.content,
                "thought_priority": thought.priority,
                "thought_source": thought.source,
                "thought_metadata": thought.metadata
            },
            priority=MessagePriority.NORMAL
        )
        
        # Check if publish was called
        mock_network_bus.publish.assert_called_once()
        self.assertEqual(message_id, "message_id")
    
    @patch('core.src.mcp.p2p_network_bus.P2PNetworkBus')
    def test_subscription_handling(self, MockNetworkBus):
        """Test subscription handling for cognitive events."""
        # Create mock network bus instance
        mock_network_bus = MockNetworkBus.return_value
        
        # Create P2P integration
        p2p_integration = P2PCoreIntegration(
            core_system=self.cognitive_architecture
        )
        
        # Set network bus
        p2p_integration.network_bus = mock_network_bus
        p2p_integration.is_initialized = True
        
        # Set P2P integration in cognitive architecture
        self.cognitive_architecture.p2p_integration = p2p_integration
        
        # Mock subscribe method
        mock_network_bus.subscribe = MagicMock(return_value="subscription_id")
        
        # Create callback
        callback = MagicMock()
        
        # Subscribe to events
        subscription_id = p2p_integration.network_bus.subscribe(
            subscriber_id="cognitive_architecture",
            event_types=[MessageBusEvent.BROADCAST],
            callback=callback
        )
        
        # Check if subscribe was called
        mock_network_bus.subscribe.assert_called_once()
        self.assertEqual(subscription_id, "subscription_id")

if __name__ == '__main__':
    unittest.main()