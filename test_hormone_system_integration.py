"""
Test suite for the HormoneSystemIntegration.

This file contains tests for the HormoneSystemIntegration class, verifying
its integration with the BrainStateAggregator and event handling.
"""

import unittest
import time
from src.mcp.hormone_system_integration import HormoneSystemIntegration

class TestHormoneSystemIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.integration = HormoneSystemIntegration()
        
    def tearDown(self):
        """Clean up after tests"""
        self.integration.shutdown()
        
    def test_event_handling(self):
        """Test hormone release in response to events"""
        # Get initial hormone levels
        initial_levels = self.integration.get_hormone_levels()
        
        # Emit task completion event
        self.integration.event_bus.emit("task_completed", {"importance": 0.9, "name": "Test Task"})
        
        # Update system
        self.integration.update()
        
        # Get updated hormone levels
        updated_levels = self.integration.get_hormone_levels()
        
        # Check that dopamine increased
        self.assertGreater(updated_levels["dopamine"], initial_levels["dopamine"])
        
    def test_cascade_effects(self):
        """Test hormone cascade effects"""
        # Emit error detection event with high severity
        self.integration.event_bus.emit("error_detected", {"severity": 0.9, "message": "Critical Error"})
        
        # Update system
        self.integration.update()
        
        # Get hormone levels
        hormone_levels = self.integration.get_hormone_levels()
        
        # Check that cortisol and histamine are elevated
        self.assertGreater(hormone_levels["cortisol"], 0.5)
        self.assertGreater(hormone_levels["histamine"], 0.5)
        
        # Check for cascade effects on other hormones
        self.assertGreater(hormone_levels["adrenaline"], 0.0)
        
    def test_brain_state_integration(self):
        """Test integration with brain state aggregator"""
        # Emit learning event
        self.integration.event_bus.emit("learning_event", {"importance": 0.8, "topic": "Test Topic"})
        
        # Update system
        self.integration.update()
        
        # Get brain state
        brain_state = self.integration.get_brain_state()
        
        # Check that hormone data is included in brain state
        self.assertIn("hormone", brain_state)
        self.assertIsNotNone(brain_state["hormone"])
        
    def test_receptor_adaptation(self):
        """Test receptor adaptation"""
        # Get initial sensitivity
        initial_sensitivity = self.integration.hormone_controller.lobes["memory"].receptor_sensitivity["dopamine"]
        
        # Adapt with positive performance
        self.integration.adapt_receptor_sensitivity("memory", "dopamine", 0.9)
        
        # Check increased sensitivity
        new_sensitivity = self.integration.hormone_controller.lobes["memory"].receptor_sensitivity["dopamine"]
        self.assertGreater(new_sensitivity, initial_sensitivity)
        
    def test_optimal_hormone_profiles(self):
        """Test learning optimal hormone profiles"""
        # Get profile for analytical task
        analytical_profile = self.integration.learn_optimal_hormone_profiles({
            "task_type": "analytical",
            "cognitive_load": "high"
        })
        
        # Get profile for creative task
        creative_profile = self.integration.learn_optimal_hormone_profiles({
            "task_type": "creative",
            "priority": "high"
        })
        
        # Check that profiles are different
        self.assertNotEqual(analytical_profile["dopamine"], creative_profile["dopamine"])
        self.assertNotEqual(analytical_profile["acetylcholine"], creative_profile["acetylcholine"])
        
if __name__ == "__main__":
    unittest.main()