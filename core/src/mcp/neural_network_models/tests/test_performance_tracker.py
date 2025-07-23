"""
Unit tests for the PerformanceTracker class.

This module contains tests for the performance tracking functionality
for neural network models, including metrics collection, history storage,
and comparative analysis.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.mcp.neural_network_models.performance_tracker import (
    PerformanceTracker,
    PerformanceMetrics,
    PerformanceComparison
)


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for the PerformanceTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for metrics
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = PerformanceTracker(metrics_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_record_metrics(self):
        """Test recording performance metrics."""
        # Record metrics for a neural implementation
        result = self.tracker.record_metrics(
            function_name="test_function",
            implementation_type="neural",
            accuracy=0.85,
            latency=50.0,
            resource_usage=0.3,
            context={"batch_size": 32},
            additional_metrics={"throughput": 100.0}
        )
        
        # Check result
        self.assertTrue(result)
        
        # Check that metrics were stored
        metrics = self.tracker.get_current_metrics("test_function", "neural")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.function_name, "test_function")
        self.assertEqual(metrics.implementation_type, "neural")
        self.assertEqual(metrics.accuracy, 0.85)
        self.assertEqual(metrics.latency, 50.0)
        self.assertEqual(metrics.resource_usage, 0.3)
        self.assertEqual(metrics.context, {"batch_size": 32})
        self.assertEqual(metrics.additional_metrics, {"throughput": 100.0})
        
    def test_record_both_implementations(self):
        """Test recording metrics for both neural and algorithmic implementations."""
        # Record metrics for neural implementation
        self.tracker.record_metrics(
            function_name="test_function",
            implementation_type="neural",
            accuracy=0.85,
            latency=50.0,
            resource_usage=0.3
        )
        
        # Record metrics for algorithmic implementation
        self.tracker.record_metrics(
            function_name="test_function",
            implementation_type="algorithmic",
            accuracy=0.80,
            latency=20.0,
            resource_usage=0.2
        )
        
        # Check that both metrics were stored
        neural_metrics = self.tracker.get_current_metrics("test_function", "neural")
        algo_metrics = self.tracker.get_current_metrics("test_function", "algorithmic")
        
        self.assertIsNotNone(neural_metrics)
        self.assertIsNotNone(algo_metrics)
        
        # Check that comparison was performed
        comparisons = self.tracker.get_comparison_history("test_function")
        self.assertEqual(len(comparisons), 1)
        
        # Check comparison result
        comparison = comparisons[0]
        self.assertEqual(comparison.function_name, "test_function")
        self.assertEqual(comparison.neural_metrics["accuracy"], 0.85)
        self.assertEqual(comparison.algorithmic_metrics["accuracy"], 0.80)
        
    def test_metrics_history(self):
        """Test storing and retrieving metrics history."""
        # Record metrics multiple times
        for i in range(5):
            self.tracker.record_metrics(
                function_name="test_function",
                implementation_type="neural",
                accuracy=0.80 + i * 0.01,
                latency=50.0 - i * 2.0,
                resource_usage=0.3
            )
            
        # Get history
        history = self.tracker.get_metrics_history("test_function", "neural")
        
        # Check history
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0].accuracy, 0.80)
        self.assertEqual(history[-1].accuracy, 0.84)
        
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        # Record metrics with improving accuracy
        for i in range(5):
            self.tracker.record_metrics(
                function_name="test_function",
                implementation_type="neural",
                accuracy=0.80 + i * 0.05,
                latency=50.0,
                resource_usage=0.3
            )
            
        # Get trend analysis
        trend = self.tracker.get_trend_analysis(
            function_name="test_function",
            implementation_type="neural",
            metric_name="accuracy"
        )
        
        # Check trend
        self.assertEqual(trend["direction"], "increasing")
        self.assertAlmostEqual(trend["magnitude"], 0.20, places=2)
        
    def test_baseline_and_anomaly_detection(self):
        """Test baseline setting and anomaly detection."""
        # Record normal metrics
        for i in range(5):
            self.tracker.record_metrics(
                function_name="test_function",
                implementation_type="neural",
                accuracy=0.85,
                latency=50.0,
                resource_usage=0.3
            )
            
        # Calculate baseline
        self.tracker.calculate_baseline_from_history("test_function", "neural")
        
        # Set anomaly threshold
        self.tracker.set_anomaly_threshold("test_function", "neural", "accuracy", 0.1)
        
        # Record anomalous metrics (accuracy drop)
        with patch.object(self.tracker.logger, 'warning') as mock_warning:
            self.tracker.record_metrics(
                function_name="test_function",
                implementation_type="neural",
                accuracy=0.70,  # Significant drop
                latency=50.0,
                resource_usage=0.3
            )
            
            # Check that warning was logged
            mock_warning.assert_called()
            
    def test_save_and_load_metrics(self):
        """Test saving and loading metrics to/from disk."""
        # Record metrics
        self.tracker.record_metrics(
            function_name="test_function",
            implementation_type="neural",
            accuracy=0.85,
            latency=50.0,
            resource_usage=0.3
        )
        
        # Save metrics
        self.tracker.save_metrics("test_function")
        
        # Create a new tracker
        new_tracker = PerformanceTracker(metrics_dir=self.temp_dir)
        
        # Load metrics
        new_tracker.load_metrics("test_function")
        
        # Check that metrics were loaded
        metrics = new_tracker.get_current_metrics("test_function", "neural")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.accuracy, 0.85)
        
    def test_recommended_implementation(self):
        """Test getting recommended implementation."""
        # Record metrics where neural is better
        self.tracker.record_metrics(
            function_name="neural_better",
            implementation_type="neural",
            accuracy=0.95,
            latency=50.0,
            resource_usage=0.3
        )
        
        self.tracker.record_metrics(
            function_name="neural_better",
            implementation_type="algorithmic",
            accuracy=0.80,
            latency=40.0,
            resource_usage=0.2
        )
        
        # Record metrics where algorithmic is better
        self.tracker.record_metrics(
            function_name="algo_better",
            implementation_type="neural",
            accuracy=0.85,
            latency=100.0,
            resource_usage=0.5
        )
        
        self.tracker.record_metrics(
            function_name="algo_better",
            implementation_type="algorithmic",
            accuracy=0.83,
            latency=20.0,
            resource_usage=0.1
        )
        
        # Check recommendations
        neural_rec, neural_reason = self.tracker.get_recommended_implementation("neural_better")
        algo_rec, algo_reason = self.tracker.get_recommended_implementation("algo_better")
        
        self.assertEqual(neural_rec, "neural")
        self.assertEqual(algo_rec, "algorithmic")
        
    def test_performance_summary(self):
        """Test getting performance summary."""
        # Record metrics for multiple functions
        self.tracker.record_metrics(
            function_name="function1",
            implementation_type="neural",
            accuracy=0.85,
            latency=50.0,
            resource_usage=0.3
        )
        
        self.tracker.record_metrics(
            function_name="function1",
            implementation_type="algorithmic",
            accuracy=0.80,
            latency=20.0,
            resource_usage=0.2
        )
        
        self.tracker.record_metrics(
            function_name="function2",
            implementation_type="neural",
            accuracy=0.90,
            latency=60.0,
            resource_usage=0.4
        )
        
        # Get summary
        summary = self.tracker.get_performance_summary()
        
        # Check summary
        self.assertIn("function1", summary["functions"])
        self.assertIn("function2", summary["functions"])
        self.assertIn("neural", summary["functions"]["function1"]["implementations"])
        self.assertIn("algorithmic", summary["functions"]["function1"]["implementations"])
        self.assertIn("neural", summary["functions"]["function2"]["implementations"])


if __name__ == '__main__':
    unittest.main()