"""
Test suite for the HormoneSystemController.

This file contains tests for the HormoneSystemController class, verifying
its hormone production, circulation, receptor adaptation, and cascade effects.
"""

import unittest
import time
from src.mcp.hormone_system_controller import HormoneSystemController

class MockEventBus:
    """Mock event bus for testing"""
    
    def __init__(self):
        self.events = []
        
    def emit(self, event_type, data):
        self.events.append((event_type, data))

class TestHormoneSystemController(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.event_bus = MockEventBus()
        self.controller = HormoneSystemController(event_bus=self.event_bus, decay_interval=0.1)
        
        # Register test lobes
        self.controller.register_lobe("task_management", position=(0, 0, 0))
        self.controller.register_lobe("memory", position=(1, 0, 0))
        self.controller.register_lobe("pattern_recognition", position=(0, 1, 0))
        self.controller.register_lobe("decision_making", position=(1, 1, 0))
        
        # Connect lobes
        self.controller.lobes["task_management"].connected_lobes = ["memory", "decision_making"]
        self.controller.lobes["memory"].connected_lobes = ["task_management", "pattern_recognition"]
        self.controller.lobes["pattern_recognition"].connected_lobes = ["memory", "decision_making"]
        self.controller.lobes["decision_making"].connected_lobes = ["task_management", "pattern_recognition"]
        
    def tearDown(self):
        """Clean up after tests"""
        self.controller.stop()
        
    def test_hormone_release(self):
        """Test hormone release and circulation"""
        # Release dopamine from task_management
        self.controller.release_hormone("task_management", "dopamine", 0.8)
        
        # Check local hormone level in source lobe
        task_levels = self.controller.get_lobe_levels("task_management")
        self.assertGreater(task_levels["dopamine"], 0.0)
        
        # Check global hormone level
        global_levels = self.controller.get_levels()
        self.assertGreater(global_levels["dopamine"], 0.0)
        
        # Check connected lobes
        memory_levels = self.controller.get_lobe_levels("memory")
        self.assertGreater(memory_levels["dopamine"], 0.0)
        
        # Check event emission
        self.assertTrue(any(event[0] == "hormone_update" for event in self.event_bus.events))
        
    def test_hormone_decay(self):
        """Test hormone decay over time"""
        # Release cortisol
        self.controller.release_hormone("task_management", "cortisol", 1.0)
        
        # Get initial level
        initial_level = self.controller.get_levels()["cortisol"]
        
        # Wait for decay
        time.sleep(0.3)
        
        # Check decayed level
        decayed_level = self.controller.get_levels()["cortisol"]
        self.assertLess(decayed_level, initial_level)
        
    def test_receptor_adaptation(self):
        """Test receptor adaptation based on performance"""
        # Get initial sensitivity
        initial_sensitivity = self.controller.lobes["memory"].receptor_sensitivity["dopamine"]
        
        # Adapt with positive performance
        self.controller.adapt_receptor_sensitivity("memory", "dopamine", 0.8)
        
        # Check increased sensitivity
        new_sensitivity = self.controller.lobes["memory"].receptor_sensitivity["dopamine"]
        self.assertGreater(new_sensitivity, initial_sensitivity)
        
        # Adapt with negative performance
        self.controller.adapt_receptor_sensitivity("memory", "dopamine", -0.8)
        
        # Check decreased sensitivity
        final_sensitivity = self.controller.lobes["memory"].receptor_sensitivity["dopamine"]
        self.assertLess(final_sensitivity, new_sensitivity)
        
    def test_hormone_cascades(self):
        """Test hormone cascade effects"""
        # Trigger stress cascade
        self.controller.global_hormone_levels["cortisol"] = 0.8
        self.controller.global_hormone_levels["adrenaline"] = 0.6
        
        # Process cascades
        result = self.controller.process_hormone_cascades()
        
        # Check cascade was triggered
        self.assertIn("stress_cascade", result.triggered_cascades)
        
        # Check affected lobes
        self.assertIn("task_management", result.affected_lobes)
        
    def test_optimal_hormone_profiles(self):
        """Test learning optimal hormone profiles for contexts"""
        # Get profile for creative task
        creative_profile = self.controller.learn_optimal_hormone_profiles({
            "task_type": "creative",
            "priority": "high"
        })
        
        # Check profile values
        self.assertGreater(creative_profile["dopamine"], 0.5)
        self.assertGreater(creative_profile["testosterone"], 0.5)
        
        # Get profile for analytical task
        analytical_profile = self.controller.learn_optimal_hormone_profiles({
            "task_type": "analytical",
            "cognitive_load": "high"
        })
        
        # Check profile values
        self.assertGreater(analytical_profile["acetylcholine"], 0.5)
        self.assertGreater(analytical_profile["norepinephrine"], 0.5)
        
if __name__ == "__main__":
    unittest.main()