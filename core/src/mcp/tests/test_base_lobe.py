"""
Tests for the BaseLobe implementation.

βtest_base_lobe(functionality_validation)
"""

import unittest
from unittest.mock import MagicMock, patch
import uuid

from core.src.mcp.lobes.base_lobe import BaseLobe
from core.src.mcp.exceptions import MCPLobeError


class TestBaseLobe(unittest.TestCase):
    """Test cases for the BaseLobe class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lobe = BaseLobe(lobe_id="test-lobe-id", name="TestLobe")
    
    def test_initialization(self):
        """Test lobe initialization."""
        self.assertEqual(self.lobe.lobe_id, "test-lobe-id")
        self.assertEqual(self.lobe.name, "TestLobe")
        self.assertEqual(self.lobe.status, "initialized")
        self.assertEqual(len(self.lobe.connections), 0)
    
    def test_initialize_with_config(self):
        """Test initializing with configuration."""
        config = {"key": "value"}
        result = self.lobe.initialize(config)
        self.assertTrue(result)
        self.assertEqual(self.lobe.config, config)
        self.assertEqual(self.lobe.status, "ready")
    
    def test_get_status(self):
        """Test getting lobe status."""
        status = self.lobe.get_status()
        self.assertEqual(status["lobe_id"], "test-lobe-id")
        self.assertEqual(status["name"], "TestLobe")
        self.assertEqual(status["status"], "initialized")
        self.assertEqual(status["connections"], 0)
    
    def test_connect_disconnect(self):
        """Test connecting and disconnecting lobes."""
        mock_lobe = MagicMock()
        mock_lobe.get_status.return_value = {
            "lobe_id": "other-lobe-id",
            "name": "OtherLobe"
        }
        
        # Test connection
        result = self.lobe.connect(mock_lobe, "test-connection")
        self.assertTrue(result)
        self.assertEqual(len(self.lobe.connections), 1)
        self.assertIn("other-lobe-id", self.lobe.connections)
        
        # Test disconnection
        result = self.lobe.disconnect(mock_lobe)
        self.assertTrue(result)
        self.assertEqual(len(self.lobe.connections), 0)
    
    def test_reset(self):
        """Test resetting the lobe."""
        # Add a connection
        mock_lobe = MagicMock()
        mock_lobe.get_status.return_value = {
            "lobe_id": "other-lobe-id",
            "name": "OtherLobe"
        }
        self.lobe.connect(mock_lobe, "test-connection")
        
        # Reset the lobe
        result = self.lobe.reset()
        self.assertTrue(result)
        
        # Check that connections are preserved
        self.assertEqual(len(self.lobe.connections), 1)
        self.assertIn("other-lobe-id", self.lobe.connections)
    
    def test_shutdown(self):
        """Test shutting down the lobe."""
        # Add a connection
        mock_lobe = MagicMock()
        mock_lobe.get_status.return_value = {
            "lobe_id": "other-lobe-id",
            "name": "OtherLobe"
        }
        self.lobe.connect(mock_lobe, "test-connection")
        
        # Shutdown the lobe
        result = self.lobe.shutdown()
        self.assertTrue(result)
        self.assertEqual(self.lobe.status, "shutdown")
        self.assertEqual(len(self.lobe.connections), 0)


if __name__ == "__main__":
    unittest.main()