"""
Test script for the Core System Infrastructure.

This script tests the functionality of the CoreSystemInfrastructure class,
including lobe registration, hormone communication, and event handling.
"""

import logging
import time
import unittest
from typing import Dict, Any, List

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from src.mcp.core_system_infrastructure import CoreSystemInfrastructure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestCoreInfrastructure")


class MockLobe:
    """Mock lobe for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.update_count = 0
        self.received_events = []
        self.context_packages = []
        
    def update(self, context_package: Dict[str, Any]):
        """Update method called by the core system."""
        self.update_count += 1
        self.context_packages.append(context_package)
        
    def register_event_handlers(self, event_bus):
        """Register event handlers with the event bus."""
        event_bus.subscribe(f"{self.name}_event", self.handle_event)
        
    def handle_event(self, event_data):
        """Handle an event."""
        self.received_events.append(event_data)


class TestCoreSystemInfrastructure(unittest.TestCase):
    """Test cases for CoreSystemInfrastructure."""
    
    def setUp(self):
        """Set up the test environment."""
        self.core_system = CoreSystemInfrastructure()
        
        # Create mock lobes
        self.mock_lobes = {
            "task_management": MockLobe("task_management"),
            "memory": MockLobe("memory"),
            "pattern_recognition": MockLobe("pattern_recognition"),
            "decision_making": MockLobe("decision_making")
        }
        
        # Register mock lobes
        for name, lobe in self.mock_lobes.items():
            self.core_system.register_lobe(
                name=name,
                instance=lobe,
                position=(0, 0, 0),
                connected_lobes=list(self.mock_lobes.keys()),
                is_left_hemisphere=(name in ["task_management", "decision_making"]),
                is_experimental=False,
                capabilities={"test"},
                hormone_receptors={"dopamine": 0.8, "serotonin": 0.7}
            )
    
    def tearDown(self):
        """Clean up after the test."""
        if self.core_system.running:
            self.core_system.stop()
    
    def test_lobe_registration(self):
        """Test lobe registration."""
        # Check that lobes were registered
        self.assertEqual(len(self.core_system.lobes), 4)
        
        # Check that lobes were registered with the hormone controller
        for name in self.mock_lobes:
            self.assertIn(name, self.core_system.hormone_controller.lobes)
    
    def test_start_stop(self):
        """Test starting and stopping the core system."""
        # Start the system
        self.core_system.start()
        self.assertTrue(self.core_system.running)
        
        # Wait for a bit to let the system run
        time.sleep(1)
        
        # Stop the system
        self.core_system.stop()
        self.assertFalse(self.core_system.running)
    
    def test_hormone_release(self):
        """Test hormone release."""
        # Start the system
        self.core_system.start()
        
        # Release a hormone
        self.core_system.release_hormone("task_management", "dopamine", 0.8)
        
        # Wait for a bit to let the hormone circulate
        time.sleep(1)
        
        # Check hormone levels
        hormone_levels = self.core_system.get_hormone_levels()
        self.assertGreater(hormone_levels["dopamine"], 0)
    
    def test_event_emission(self):
        """Test event emission."""
        # Start the system
        self.core_system.start()
        
        # Emit an event
        self.core_system.emit_event(
            "task_management_event",
            {"test": "data"},
            signal_type="excitatory",
            context={"source": "test"}
        )
        
        # Wait for a bit to let the event be processed
        time.sleep(1)
        
        # Check that the event was received
        self.assertEqual(len(self.mock_lobes["task_management"].received_events), 1)
        self.assertEqual(self.mock_lobes["task_management"].received_events[0]["data"]["test"], "data")
    
    def test_lobe_update(self):
        """Test lobe updates."""
        # Start the system
        self.core_system.start()
        
        # Wait for a bit to let the lobes be updated
        time.sleep(1)
        
        # Check that lobes were updated
        for name, lobe in self.mock_lobes.items():
            self.assertGreater(lobe.update_count, 0)
            self.assertGreater(len(lobe.context_packages), 0)
    
    def test_hemisphere_classification(self):
        """Test hemisphere classification."""
        # Get lobes by hemisphere
        left_lobes = self.core_system.get_lobes_by_hemisphere(left_hemisphere=True)
        right_lobes = self.core_system.get_lobes_by_hemisphere(left_hemisphere=False)
        
        # Check that lobes were classified correctly
        self.assertEqual(len(left_lobes), 2)
        self.assertEqual(len(right_lobes), 2)
        self.assertIn("task_management", left_lobes)
        self.assertIn("decision_making", left_lobes)
        self.assertIn("memory", right_lobes)
        self.assertIn("pattern_recognition", right_lobes)
    
    def test_capabilities(self):
        """Test capability tracking."""
        # Get capabilities
        capabilities = self.core_system.get_lobe_capabilities()
        
        # Check that capabilities were tracked correctly
        self.assertIn("test", capabilities)
        self.assertEqual(len(capabilities["test"]), 4)


if __name__ == "__main__":
    unittest.main()