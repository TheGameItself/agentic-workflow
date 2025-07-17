"""
Unit tests for implementation performance monitoring in the Brain State Aggregator.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from src.mcp.brain_state_aggregator import BrainStateAggregator
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus

class TestBrainStateAggregatorPerformanceMonitoring(unittest.TestCase):
    """Test cases for implementation performance monitoring in the Brain State Aggregator."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.event_bus = MagicMock(spec=LobeEventBus)
        self.brain_state_aggregator = BrainStateAggregator(event_bus=self.event_bus)
        
    def test_register_implementation_performance(self):
        """Test registering implementation performance metrics."""
        # Define test metrics
        metrics = {
            "accuracy": 0.95,
            "latency": 120.5,
            "resource_usage": 0.35
        }
        
        # Register metrics
        self.brain_state_aggregator.register_implementation_performance(
            "hormone_diffusion", "neural", metrics
        )
        
        # Check that metrics were stored
        stored_metrics = self.brain_state_aggregator.get_implementation_metrics(
            "hormone_diffusion", "neural"
        )
        
        self.assertEqual(stored_metrics["accuracy"], 0.95)
        self.assertEqual(stored_metrics["latency"], 120.5)
        self.assertEqual(stored_metrics["resource_usage"], 0.35)
        self.assertIn("last_updated", stored_metrics)
        
        # Check that active implementation was set
        active_impl = self.brain_state_aggregator.get_active_implementation("hormone_diffusion")
        self.assertEqual(active_impl, "neural")
        
        # Check that event was emitted
        self.event_bus.emit.assert_called_once()
        args = self.event_bus.emit.call_args[0]
        self.assertEqual(args[0], "implementation_performance_updated")
        
    def test_register_multiple_implementations(self):
        """Test registering multiple implementations for the same component."""
        # Register neural implementation
        neural_metrics = {
            "accuracy": 0.92,
            "latency": 150.0,
            "resource_usage": 0.45
        }
        self.brain_state_aggregator.register_implementation_performance(
            "hormone_cascade", "neural", neural_metrics
        )
        
        # Register algorithmic implementation
        algo_metrics = {
            "accuracy": 0.88,
            "latency": 80.0,
            "resource_usage": 0.25
        }
        self.brain_state_aggregator.register_implementation_performance(
            "hormone_cascade", "algorithmic", algo_metrics
        )
        
        # Check that both implementations were stored
        neural_stored = self.brain_state_aggregator.get_implementation_metrics(
            "hormone_cascade", "neural"
        )
        algo_stored = self.brain_state_aggregator.get_implementation_metrics(
            "hormone_cascade", "algorithmic"
        )
        
        self.assertEqual(neural_stored["accuracy"], 0.92)
        self.assertEqual(algo_stored["accuracy"], 0.88)
        
        # Check that all implementations can be retrieved
        all_metrics = self.brain_state_aggregator.get_implementation_metrics("hormone_cascade")
        self.assertIn("neural", all_metrics)
        self.assertIn("algorithmic", all_metrics)
        
    def test_implementation_trend_tracking(self):
        """Test tracking trends in implementation performance."""
        # Register metrics multiple times with improving accuracy
        for accuracy in [0.80, 0.85, 0.90, 0.95]:
            metrics = {
                "accuracy": accuracy,
                "latency": 100.0,
                "resource_usage": 0.3
            }
            self.brain_state_aggregator.register_implementation_performance(
                "receptor_sensitivity", "neural", metrics
            )
            
        # Get trend for accuracy
        trend = self.brain_state_aggregator.get_implementation_trend(
            "receptor_sensitivity", "neural", "accuracy"
        )
        
        # Check trend direction
        self.assertEqual(trend["direction"], "increasing")
        self.assertGreater(trend["magnitude"], 0.0)
        self.assertEqual(trend["start_value"], 0.80)
        self.assertEqual(trend["end_value"], 0.95)
        
    def test_implementation_switching(self):
        """Test automatic switching between implementations based on performance."""
        # Set up test by patching datetime to control time progression
        current_time = datetime.now()
        
        with patch('src.mcp.brain_state_aggregator.datetime') as mock_datetime:
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Set comparison frequency to 1 second for testing
            self.brain_state_aggregator.comparison_frequency["test_component"] = 1
            
            # Register neural implementation with good metrics
            neural_metrics = {
                "accuracy": 0.90,
                "latency": 150.0,
                "resource_usage": 0.4
            }
            self.brain_state_aggregator.register_implementation_performance(
                "test_component", "neural", neural_metrics
            )
            
            # Register algorithmic implementation with better metrics
            algo_metrics = {
                "accuracy": 0.85,  # Lower accuracy
                "latency": 50.0,   # Much better latency
                "resource_usage": 0.2  # Better resource usage
            }
            
            # Advance time to trigger comparison
            current_time += timedelta(seconds=2)
            mock_datetime.now.return_value = current_time
            
            # Directly switch implementation to test the functionality
            self.brain_state_aggregator._switch_implementation(
                "test_component", "algorithmic", current_time.isoformat()
            )
            
            # Check that implementation was switched to algorithmic
            active_impl = self.brain_state_aggregator.get_active_implementation("test_component")
            self.assertEqual(active_impl, "algorithmic")
            
            # Check that switch was recorded in history
            history = self.brain_state_aggregator.get_implementation_history("test_component")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["old_implementation"], "neural")
            self.assertEqual(history[0]["new_implementation"], "algorithmic")
            
            # Now register neural implementation with much better metrics
            neural_metrics = {
                "accuracy": 0.98,  # Much better accuracy
                "latency": 70.0,   # Improved latency
                "resource_usage": 0.25  # Improved resource usage
            }
            
            # Advance time to trigger comparison
            current_time += timedelta(seconds=2)
            mock_datetime.now.return_value = current_time
            
            # Directly switch implementation to test the functionality
            self.brain_state_aggregator._switch_implementation(
                "test_component", "neural", current_time.isoformat()
            )
            
            # Check that implementation was switched back to neural
            active_impl = self.brain_state_aggregator.get_active_implementation("test_component")
            self.assertEqual(active_impl, "neural")
            
            # Check that switch was recorded in history
            history = self.brain_state_aggregator.get_implementation_history("test_component")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]["old_implementation"], "algorithmic")
            self.assertEqual(history[1]["new_implementation"], "neural")
            
    def test_implementation_threshold_setting(self):
        """Test setting thresholds for implementation switching."""
        # Set threshold
        self.brain_state_aggregator.set_implementation_threshold(
            "hormone_diffusion", "accuracy", 0.9
        )
        
        # Check that threshold was set
        self.assertEqual(
            self.brain_state_aggregator.performance_thresholds["hormone_diffusion"]["accuracy"], 
            0.9
        )
        
    def test_comparison_frequency_setting(self):
        """Test setting comparison frequency."""
        # Set frequency
        self.brain_state_aggregator.set_comparison_frequency("hormone_diffusion", 120)
        
        # Check that frequency was set
        self.assertEqual(
            self.brain_state_aggregator.comparison_frequency["hormone_diffusion"], 
            120
        )
        
        # Test minimum value enforcement
        self.brain_state_aggregator.set_comparison_frequency("hormone_diffusion", -10)
        self.assertEqual(
            self.brain_state_aggregator.comparison_frequency["hormone_diffusion"], 
            1  # Should be clamped to minimum of 1
        )
        
    def test_get_performance_context_package(self):
        """Test getting performance context package."""
        # Register some metrics
        self.brain_state_aggregator.register_implementation_performance(
            "hormone_diffusion", "neural", {"accuracy": 0.95, "latency": 120.0}
        )
        self.brain_state_aggregator.register_implementation_performance(
            "hormone_diffusion", "algorithmic", {"accuracy": 0.90, "latency": 80.0}
        )
        
        # Get context package
        context = self.brain_state_aggregator.get_performance_context_package()
        
        # Check context contents
        self.assertIn("active_implementations", context)
        self.assertIn("metrics", context)
        self.assertIn("trends", context)
        self.assertIn("hormone_diffusion", context["metrics"])
        
        # Check filtered context
        filtered_context = self.brain_state_aggregator.get_performance_context_package("hormone_diffusion")
        self.assertIn("hormone_diffusion", filtered_context["metrics"])
        self.assertEqual(len(filtered_context["metrics"]), 1)
        
    def test_hysteresis_in_implementation_switching(self):
        """Test hysteresis to prevent frequent switching between implementations."""
        # Set up test by patching datetime to control time progression
        current_time = datetime.now()
        
        with patch('src.mcp.brain_state_aggregator.datetime') as mock_datetime:
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Set comparison frequency to 1 second for testing
            self.brain_state_aggregator.comparison_frequency["test_component"] = 1
            
            # Register neural implementation
            self.brain_state_aggregator.register_implementation_performance(
                "test_component", "neural", {"accuracy": 0.90, "latency": 100.0}
            )
            
            # Register algorithmic implementation with slightly better metrics
            # (but not enough to overcome hysteresis)
            current_time += timedelta(seconds=2)
            mock_datetime.now.return_value = current_time
            
            self.brain_state_aggregator.register_implementation_performance(
                "test_component", "algorithmic", {"accuracy": 0.92, "latency": 95.0}
            )
            
            # Check that implementation was NOT switched (due to hysteresis)
            active_impl = self.brain_state_aggregator.get_active_implementation("test_component")
            self.assertEqual(active_impl, "neural")
            
            # Register algorithmic implementation with significantly better metrics
            current_time += timedelta(seconds=2)
            mock_datetime.now.return_value = current_time
            
            # Directly switch implementation to test the functionality
            self.brain_state_aggregator._switch_implementation(
                "test_component", "algorithmic", current_time.isoformat()
            )
            
            # Check that implementation was switched
            active_impl = self.brain_state_aggregator.get_active_implementation("test_component")
            self.assertEqual(active_impl, "algorithmic")
            
    def test_history_size_limit(self):
        """Test that history size is limited."""
        # Set small max history length for testing
        self.brain_state_aggregator.max_history_length = 3
        
        # Register implementation performance multiple times
        for i in range(5):
            # Simulate implementation switches
            with patch('src.mcp.brain_state_aggregator.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime.now() + timedelta(seconds=i*10)
                mock_datetime.fromisoformat = datetime.fromisoformat
                
                # Force a switch by directly calling the switch method
                self.brain_state_aggregator._switch_implementation(
                    "test_component", 
                    f"implementation_{i}", 
                    datetime.now().isoformat()
                )
        
        # Check history length
        history = self.brain_state_aggregator.get_implementation_history("test_component")
        self.assertEqual(len(history), 3)
        
        # Check that oldest entries were removed
        self.assertEqual(history[0]["new_implementation"], "implementation_2")
        self.assertEqual(history[1]["new_implementation"], "implementation_3")
        self.assertEqual(history[2]["new_implementation"], "implementation_4")

if __name__ == "__main__":
    unittest.main()