#!/usr/bin/env python3
"""
Test P2P Network Integration for MCP Core System
Tests the integration of P2P network functionality with the core system.
"""

import asyncio
import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import core components
from src.mcp.core_system import MCPCoreSystem, SystemConfiguration
from src.mcp.spinal_column import SpinalColumn
from src.mcp.p2p_network_bus import P2PNetworkBus, MessageBusEvent, MessagePriority
from src.mcp.p2p_core_integration import P2PCoreIntegration
from src.mcp.core_system_p2p_integration import CoreSystemP2PIntegration, integrate_p2p_with_core_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestP2PIntegration(unittest.TestCase):
    """Test P2P network integration with core system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary data directory
        self.data_dir = Path("test_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Create core system configuration
        self.config = SystemConfiguration(
            max_workers=2,
            enable_async=True,
            enable_monitoring=True,
            log_level="INFO",
            data_directory=str(self.data_dir),
            backup_enabled=False,
            performance_optimization=False,
            experimental_features=True,
            hormone_system_enabled=True,
            vector_storage_enabled=True
        )
        
        # Create core system
        self.core_system = MCPCoreSystem(self.config)
        
        # Create spinal column
        self.spinal_column = SpinalColumn(
            input_dim=64,
            hidden_dim=128,
            output_dim=64
        )
        
        # Set spinal column in core system
        self.core_system.spinal_column = self.spinal_column
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary data directory
        import shutil
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
    
    @patch('src.mcp.p2p_network_bus.P2PNetworkBus.start')
    @patch('src.mcp.p2p_network_bus.P2PNetworkBus.stop')
    @patch('src.mcp.p2p_core_integration.P2PCoreIntegration.initialize')
    @patch('src.mcp.p2p_core_integration.P2PCoreIntegration.shutdown')
    async def test_p2p_integration_lifecycle(self, mock_shutdown, mock_initialize, mock_stop, mock_start):
        """Test P2P integration lifecycle."""
        # Mock successful initialization
        mock_initialize.return_value = True
        mock_shutdown.return_value = True
        
        # Integrate P2P with core system
        success = await integrate_p2p_with_core_system(self.core_system)
        
        # Check integration success
        self.assertTrue(success)
        self.assertTrue(hasattr(self.core_system, 'p2p_integration'))
        
        # Check if P2P is enabled
        is_enabled = CoreSystemP2PIntegration.is_p2p_enabled(self.core_system)
        self.assertTrue(is_enabled)
        
        # Get P2P status
        status = CoreSystemP2PIntegration.get_p2p_status(self.core_system)
        self.assertIsNotNone(status)
        
        # Initialize core system
        await self.core_system.initialize()
        
        # Check if initialize was called
        mock_initialize.assert_called_once()
        
        # Shutdown core system
        await self.core_system.shutdown()
        
        # Check if shutdown was called
        mock_shutdown.assert_called_once()
    
    @patch('src.mcp.p2p_network_bus.P2PNetworkBus')
    async def test_message_publishing(self, mock_network_bus):
        """Test message publishing through P2P network bus."""
        # Create P2P integration
        p2p_integration = P2PCoreIntegration(
            core_system=self.core_system,
            spinal_column=self.spinal_column
        )
        
        # Mock network bus
        p2p_integration.network_bus = mock_network_bus
        p2p_integration.is_initialized = True
        
        # Mock publish method
        mock_network_bus.publish = MagicMock(return_value="message_id")
        
        # Publish a message
        message_id = await p2p_integration.network_bus.publish(
            event_type=MessageBusEvent.STATUS_UPDATE,
            content={"status": "test"},
            priority=MessagePriority.NORMAL
        )
        
        # Check if publish was called
        mock_network_bus.publish.assert_called_once()
        self.assertEqual(message_id, "message_id")
    
    @patch('src.mcp.p2p_network_bus.P2PNetworkBus')
    async def test_subscription_handling(self, mock_network_bus):
        """Test subscription handling in P2P network bus."""
        # Create P2P integration
        p2p_integration = P2PCoreIntegration(
            core_system=self.core_system,
            spinal_column=self.spinal_column
        )
        
        # Mock network bus
        p2p_integration.network_bus = mock_network_bus
        p2p_integration.is_initialized = True
        
        # Mock subscribe method
        mock_network_bus.subscribe = MagicMock(return_value="subscription_id")
        
        # Create callback
        callback = MagicMock()
        
        # Subscribe to events
        subscription_id = p2p_integration.network_bus.subscribe(
            subscriber_id="test_subscriber",
            event_types=[MessageBusEvent.STATUS_UPDATE],
            callback=callback
        )
        
        # Check if subscribe was called
        mock_network_bus.subscribe.assert_called_once()
        self.assertEqual(subscription_id, "subscription_id")

if __name__ == '__main__':
    unittest.main()