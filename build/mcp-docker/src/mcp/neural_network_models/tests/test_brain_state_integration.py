"""
Unit tests for the NeuralPerformanceIntegration class.

This module contains tests for the integration between the PerformanceTracker
and BrainStateAggregator.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock, Mock

from src.mcp.neural_network_models.performance_tracker import PerformanceTracker
from src.mcp.neural_network_models.brain_state_integration import NeuralPerformanceIntegration


class TestNeuralPerformanceIntegration(unittest.TestCase):
    """Test cases for the NeuralPerformanceIntegration class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for metrics
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock brain state aggregator
        self.mock_brain_state = MagicMock()
        
        # Create real performance tracker
        self.tracker = PerformanceTracker(metrics_dir=self.temp_dir)
        
        # Create integration
        self.integration = NeuralPerformanceIntegration(
            brain_state_aggregator=self.mock_brain_state,
            performance_tracker=self.tracker
        )
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_report_neural_metrics(self):
        """Test reporting neural metrics."""
        # Report metrics
        result = self.integration.report_neural_metrics(
            function_name="test_function",
            accuracy=0.85,
            latency=50.0,
            resource_usage=0.3,
            additional_metrics={"throughput": 100.0},
            context={"batch_size": 32}
        )
        
        # Check result
        self.assertTrue(result)
        
        # Check that metrics were stored in tracker
        metrics = self.tracker.get_current_metrics("test_function", "neural")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.accuracy, 0.85)
        
        # Check that brain state aggregator was called
        self.mock_brain_state.register_implementation_performance.assert_called_once()
        args, kwargs = self.mock_brain_state.register_implementation_performance.call_args
        self.assertEqual(kwargs["component"], "test_function")
        self.assertEqual(kwargs["implementation_type"], "neural")
        self.assertEqual(kwargs["metrics"]["accuracy"], 0.85)
        self.assertEqual(kwargs["metrics"]["throughput"], 100.0)
        
    def test_report_algorithmic_metrics(self):
        """Test reporting algorithmic metrics."""
        # Report metrics
        result = self.integration.report_algorithmic_metrics(
            function_name="test_function",
            accuracy=0.80,
            latency=20.0,
            resource_usage=0.2
        )
        
        # Check result
        self.assertTrue(result)
        
        # Check that metrics were stored in tracker
        metrics = self.tracker.get_current_metrics("test_function", "algorithmic")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.accuracy, 0.80)
        
        # Check that brain state aggregator was called
        self.mock_brain_state.register_implementation_performance.assert_called_once()
        args, kwargs = self.mock_brain_state.register_implementation_performance.call_args
        self.assertEqual(kwargs["component"], "test_function")
        self.assertEqual(kwargs["implementation_type"], "algorithmic")
        self.assertEqual(kwargs["metrics"]["accuracy"], 0.80)
        
    def test_sync_implementation_decisions(self):
        """Test synchronizing implementation decisions."""
        # Report metrics for both implementations
        self.integration.report_neural_metrics(
            function_name="test_function",
            accuracy=0.95,  # Neural is better
            latency=50.0,
            resource_usage=0.3
        )
        
        self.integration.report_algorithmic_metrics(
            function_name="test_function",
            accuracy=0.80,
            latency=20.0,
            resource_usage=0.2
        )
        
        # Mock brain state to return algorithmic as active implementation
        self.mock_brain_state.get_active_implementation.return_value = "algorithmic"
        
        # Sync decisions
        decisions = self.integration.sync_implementation_decisions()
        
        # Check decisions
        self.assertIn("test_function", decisions)
        impl_type, reason = decisions["test_function"]
        self.assertEqual(impl_type, "neural")  # Should recommend neural
        
        # Check that brain state was updated
        self.mock_brain_state.register_implementation_performance.assert_called()
        
    def test_get_performance_history(self):
        """Test getting performance history."""
        # Report metrics multiple times
        for i in range(3):
            self.integration.report_neural_metrics(
                function_name="test_function",
                accuracy=0.80 + i * 0.05,
                latency=50.0,
                resource_usage=0.3
            )
            
        # Mock brain state history
        self.mock_brain_state.implementation_history = {
            "test_function": [
                {"timestamp": "2023-01-01T00:00:00", "old_implementation": "algorithmic", "new_implementation": "neural"}
            ]
        }
        
        # Get history
        history = self.integration.get_performance_history("test_function")
        
        # Check history
        self.assertIn("tracker_history", history)
        self.assertIn("brain_state_history", history)
        self.assertIn("neural", history["tracker_history"])
        self.assertEqual(len(history["tracker_history"]["neural"]), 3)
        self.assertEqual(len(history["brain_state_history"]), 1)
        
    def test_save_all_metrics(self):
        """Test saving all metrics."""
        # Report metrics for multiple functions
        self.integration.report_neural_metrics(
            function_name="function1",
            accuracy=0.85,
            latency=50.0,
            resource_usage=0.3
        )
        
        self.integration.report_neural_metrics(
            function_name="function2",
            accuracy=0.90,
            latency=60.0,
            resource_usage=0.4
        )
        
        # Save metrics
        result = self.integration.save_all_metrics()
        
        # Check result
        self.assertTrue(result)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "function1.metrics")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "function2.metrics")))


if __name__ == '__main__':
    unittest.main()