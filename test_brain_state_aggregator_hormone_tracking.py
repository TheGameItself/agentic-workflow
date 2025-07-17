"""
Unit tests for hormone level tracking in the Brain State Aggregator.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.mcp.brain_state_aggregator import BrainStateAggregator
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus

class TestBrainStateAggregatorHormoneTracking(unittest.TestCase):
    """Test cases for hormone level tracking in the Brain State Aggregator."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.event_bus = MagicMock(spec=LobeEventBus)
        self.hormone_engine = MagicMock()
        self.hormone_engine.get_levels.return_value = {
            "dopamine": 0.5,
            "serotonin": 0.6,
            "cortisol": 0.3
        }
        self.brain_state_aggregator = BrainStateAggregator(
            hormone_engine=self.hormone_engine,
            event_bus=self.event_bus
        )
        
    def test_update_hormone_levels(self):
        """Test updating hormone levels."""
        # Define test hormone levels
        hormone_levels = {
            "dopamine": 0.7,
            "serotonin": 0.4,
            "cortisol": 0.2
        }
        
        # Update hormone levels
        self.brain_state_aggregator.update_hormone_levels(hormone_levels)
        
        # Check that hormone levels were updated
        self.assertEqual(self.brain_state_aggregator.hormone_levels["dopamine"], 0.7)
        self.assertEqual(self.brain_state_aggregator.hormone_levels["serotonin"], 0.4)
        self.assertEqual(self.brain_state_aggregator.hormone_levels["cortisol"], 0.2)
        
        # Check that buffer was updated
        self.assertEqual(self.brain_state_aggregator.buffers["hormone"], hormone_levels)
        
        # Check that event was emitted
        self.event_bus.emit.assert_called_once()
        args = self.event_bus.emit.call_args[0]
        self.assertEqual(args[0], "hormone_levels_updated")
        
    def test_update_hormone_levels_with_source(self):
        """Test updating hormone levels with source lobe."""
        # Define test hormone levels
        hormone_levels = {
            "dopamine": 0.7,
            "serotonin": 0.4
        }
        
        # Update hormone levels with source
        self.brain_state_aggregator.update_hormone_levels(hormone_levels, source_lobe="memory")
        
        # Check that source tracking was updated
        self.assertEqual(
            self.brain_state_aggregator.hormone_source_tracking["dopamine"]["memory"], 
            0.7
        )
        self.assertEqual(
            self.brain_state_aggregator.hormone_source_tracking["serotonin"]["memory"], 
            0.4
        )
        
    def test_hormone_history(self):
        """Test hormone history tracking."""
        # Update hormone levels multiple times
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.3})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.5})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.7})
        
        # Check history length
        history = self.brain_state_aggregator.get_hormone_history("dopamine")
        self.assertEqual(len(history), 3)
        
        # Check history values
        self.assertEqual(history[0]["level"], 0.3)
        self.assertEqual(history[1]["level"], 0.5)
        self.assertEqual(history[2]["level"], 0.7)
        
        # Check delta values
        self.assertAlmostEqual(history[0]["delta"], 0.3)  # 0.3 - 0.0
        self.assertAlmostEqual(history[1]["delta"], 0.2)  # 0.5 - 0.3
        self.assertAlmostEqual(history[2]["delta"], 0.2)  # 0.7 - 0.5
        
    def test_hormone_trend(self):
        """Test hormone trend calculation."""
        # Update hormone levels with increasing trend
        self.brain_state_aggregator.update_hormone_levels({"cortisol": 0.1})
        self.brain_state_aggregator.update_hormone_levels({"cortisol": 0.2})
        self.brain_state_aggregator.update_hormone_levels({"cortisol": 0.3})
        self.brain_state_aggregator.update_hormone_levels({"cortisol": 0.4})
        
        # Get trend
        trend = self.brain_state_aggregator.get_hormone_trend("cortisol")
        
        # Check trend direction
        self.assertEqual(trend["direction"], "increasing")
        self.assertGreater(trend["magnitude"], 0.0)
        
        # Update hormone levels with decreasing trend
        self.brain_state_aggregator.update_hormone_levels({"serotonin": 0.8})
        self.brain_state_aggregator.update_hormone_levels({"serotonin": 0.7})
        self.brain_state_aggregator.update_hormone_levels({"serotonin": 0.6})
        self.brain_state_aggregator.update_hormone_levels({"serotonin": 0.5})
        
        # Get trend
        trend = self.brain_state_aggregator.get_hormone_trend("serotonin")
        
        # Check trend direction
        self.assertEqual(trend["direction"], "decreasing")
        self.assertGreater(trend["magnitude"], 0.0)
        
        # Update hormone levels with stable trend
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.5})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.51})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.49})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.5})
        
        # Get trend
        trend = self.brain_state_aggregator.get_hormone_trend("dopamine")
        
        # Check trend direction
        self.assertEqual(trend["direction"], "stable")
        self.assertLess(trend["magnitude"], 0.05)
        
    def test_hormone_thresholds(self):
        """Test setting and getting hormone thresholds."""
        # Set thresholds
        self.brain_state_aggregator.set_hormone_threshold("dopamine", "activation", 0.3)
        self.brain_state_aggregator.set_hormone_threshold("dopamine", "reward", 0.7)
        
        # Check thresholds
        self.assertEqual(
            self.brain_state_aggregator.hormone_thresholds["dopamine"]["activation"], 
            0.3
        )
        self.assertEqual(
            self.brain_state_aggregator.hormone_thresholds["dopamine"]["reward"], 
            0.7
        )
        
    def test_register_hormone_cascade(self):
        """Test registering hormone cascades."""
        # Define cascade data
        cascade_data = {
            "name": "stress_cascade",
            "trigger": "cortisol",
            "affected_hormones": ["adrenaline", "norepinephrine"],
            "magnitude": 0.8
        }
        
        # Register cascade
        self.brain_state_aggregator.register_hormone_cascade(cascade_data)
        
        # Check cascade history
        self.assertEqual(len(self.brain_state_aggregator.hormone_cascade_history), 1)
        self.assertEqual(
            self.brain_state_aggregator.hormone_cascade_history[0]["name"], 
            "stress_cascade"
        )
        
    def test_get_hormone_context_package(self):
        """Test getting hormone context package."""
        # Set up test data
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.7, "serotonin": 0.5})
        self.brain_state_aggregator.set_hormone_threshold("dopamine", "activation", 0.3)
        
        # Get context package
        context = self.brain_state_aggregator.get_hormone_context_package("memory")
        
        # Check context contents
        self.assertIn("current_levels", context)
        self.assertIn("trends", context)
        self.assertIn("thresholds", context)
        self.assertIn("recent_cascades", context)
        
        # Check hormone levels in context
        self.assertEqual(context["current_levels"]["dopamine"], 0.7)
        self.assertEqual(context["current_levels"]["serotonin"], 0.5)
        
        # Check thresholds in context
        self.assertEqual(context["thresholds"]["dopamine"]["activation"], 0.3)
        
    def test_update_buffers_with_hormone_engine(self):
        """Test updating buffers with hormone engine."""
        # Set up mock hormone engine
        self.hormone_engine.get_levels.return_value = {
            "dopamine": 0.6,
            "serotonin": 0.7
        }
        
        # Add get_recent_cascades method to mock
        self.hormone_engine.get_recent_cascades = MagicMock()
        self.hormone_engine.get_recent_cascades.return_value = [
            {
                "name": "reward_cascade",
                "trigger": "dopamine",
                "affected_hormones": ["serotonin", "oxytocin"]
            }
        ]
        
        # Update buffers
        self.brain_state_aggregator.update_buffers()
        
        # Check hormone levels
        self.assertEqual(self.brain_state_aggregator.hormone_levels["dopamine"], 0.6)
        self.assertEqual(self.brain_state_aggregator.hormone_levels["serotonin"], 0.7)
        
        # Check cascade history
        self.assertEqual(len(self.brain_state_aggregator.hormone_cascade_history), 1)
        self.assertEqual(
            self.brain_state_aggregator.hormone_cascade_history[0]["name"], 
            "reward_cascade"
        )
        
    def test_hormone_level_limits(self):
        """Test that hormone levels are limited to 0.0-1.0 range."""
        # Try to set invalid hormone levels
        self.brain_state_aggregator.update_hormone_levels({
            "dopamine": 1.5,  # Above maximum
            "serotonin": -0.3  # Below minimum
        })
        
        # Check that levels were clamped to valid range
        self.assertEqual(self.brain_state_aggregator.hormone_levels["dopamine"], 1.0)
        self.assertEqual(self.brain_state_aggregator.hormone_levels["serotonin"], 0.0)
        
    def test_history_size_limit(self):
        """Test that history size is limited."""
        # Set small max history length for testing
        self.brain_state_aggregator.max_history_length = 3
        
        # Update hormone levels multiple times
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.1})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.2})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.3})
        self.brain_state_aggregator.update_hormone_levels({"dopamine": 0.4})
        
        # Check history length
        history = self.brain_state_aggregator.get_hormone_history("dopamine")
        self.assertEqual(len(history), 3)
        
        # Check that oldest entry was removed
        self.assertEqual(history[0]["level"], 0.2)
        self.assertEqual(history[1]["level"], 0.3)
        self.assertEqual(history[2]["level"], 0.4)

if __name__ == "__main__":
    unittest.main()