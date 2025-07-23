#!/usr/bin/env python3
"""
Debug test for P2P integration with cognitive architecture.
"""

import asyncio
import logging
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestP2PIntegration(unittest.TestCase):
    """Test P2P integration with mocked components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mocks
        self.mock_core_system = MagicMock()
        self.mock_spinal_column = MagicMock()
        
        # Set up core system mock
        self.mock_core_system.config = MagicMock()
        self.mock_core_system.config.data_directory = "test_data"
        self.mock_core_system.config.experimental_features = True
        
        # Set up metrics mock
        self.mock_core_system.metrics = MagicMock()
        self.mock_core_system.metrics.cpu_usage = 0.5
        self.mock_core_system.metrics.memory_usage = 0.3
        self.mock_core_system.metrics.active_lobes = 5
        self.mock_core_system.metrics.uptime = 3600
        self.mock_core_system.metrics.last_updated = MagicMock()
        self.mock_core_system.metrics.last_updated.timestamp.return_value = 1234567890
        
        # Set spinal column in core system
        self.mock_core_system.spinal_column = self.mock_spinal_column
        
        # Mock core system methods
        self.mock_core_system._monitoring_loop = MagicMock()
        self.mock_core_system.shutdown = MagicMock()
        self.mock_core_system.get_status = MagicMock(return_value={})
        self.mock_core_system._shutdown_event = MagicMock()
        self.mock_core_system._shutdown_event.is_set = MagicMock(return_value=False)
    
    @patch('core.src.mcp.p2p_network_bus.P2PNetworkBus')
    @patch('core.src.mcp.p2p_core_integration.P2PCoreIntegration')
    def test_p2p_integration(self, MockP2PCoreIntegration, MockNetworkBus):
        """Test P2P integration with mocked components."""
        # Set up mocks
        mock_p2p_integration = MockP2PCoreIntegration.return_value
        mock_network_bus = MockNetworkBus.return_value
        
        # Set up network bus in p2p integration
        mock_p2p_integration.network_bus = mock_network_bus
        mock_p2p_integration.is_initialized = True
        
        # Set up p2p integration in core system
        self.mock_core_system.p2p_integration = mock_p2p_integration
        
        # Test if P2P is enabled
        from core.src.mcp.core_system_p2p_integration import CoreSystemP2PIntegration
        is_enabled = CoreSystemP2PIntegration.is_p2p_enabled(self.mock_core_system)
        self.assertTrue(is_enabled)
        
        # Test get P2P status
        status = CoreSystemP2PIntegration.get_p2p_status(self.mock_core_system)
        self.assertIsNotNone(status)
        
        # Test enhanced monitoring loop
        monitoring_loop = self.mock_core_system._monitoring_loop
        monitoring_loop()
        
        # Test enhanced shutdown
        shutdown = self.mock_core_system.shutdown
        shutdown()

if __name__ == '__main__':
    unittest.main()