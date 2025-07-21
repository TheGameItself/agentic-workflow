"""
Tests for the MonitoringSystem visualization and reporting capabilities.

This module contains tests for the visualization data generation, performance comparison
reporting, and trend analysis features of the MonitoringSystem.
"""

import unittest
import os
import sqlite3
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.mcp.monitoring_system_visualization import (
    MonitoringSystemVisualization,
    VisualizationOptions,
    PerformanceComparisonReport
)
from src.mcp.monitoring_system_visualization_integration import MonitoringSystemWithVisualization


class TestMonitoringVisualization(unittest.TestCase):
    """Test suite for monitoring system visualization capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary database for testing
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self.visualization = MonitoringSystemVisualization(self.temp_db_path)
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test environment."""
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)
    
    def _create_test_data(self):
        """Create test data in the temporary database."""
        with sqlite3.connect(self.temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Create required tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hormone_levels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    hormone_name TEXT NOT NULL,
                    level REAL NOT NULL,
                    source_lobe TEXT,
                    context TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hormone_cascades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cascade_name TEXT NOT NULL,
                    trigger_hormone TEXT NOT NULL,
                    trigger_level REAL NOT NULL,
                    affected_hormones TEXT NOT NULL,
                    affected_lobes TEXT NOT NULL,
                    duration REAL,
                    effects TEXT,
                    feedback_loops TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    implementation TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS implementation_switches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    old_implementation TEXT NOT NULL,
                    new_implementation TEXT NOT NULL,
                    switch_timestamp TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    performance_comparison TEXT NOT NULL,
                    switch_trigger TEXT NOT NULL,
                    confidence_score REAL,
                    expected_improvement TEXT
                )
            """)
            
            # Insert test hormone levels
            now = datetime.now()
            hormones = ["dopamine", "serotonin", "cortisol"]
            lobes = ["task_management", "decision_making", "memory"]
            
            for i in range(100):
                timestamp = (now - timedelta(minutes=i)).isoformat()
                
                for hormone in hormones:
                    # Create some patterns in the data
                    if hormone == "dopamine":
                        level = 0.7 + 0.2 * (i % 10) / 10.0  # Oscillating pattern
                    elif hormone == "serotonin":
                        level = 0.5 + 0.01 * i  # Increasing trend
                    else:  # cortisol
                        level = 0.8 - 0.005 * i  # Decreasing trend
                    
                    source_lobe = lobes[i % len(lobes)]
                    
                    cursor.execute("""
                        INSERT INTO hormone_levels 
                        (timestamp, hormone_name, level, source_lobe)
                        VALUES (?, ?, ?, ?)
                    """, (timestamp, hormone, level, source_lobe))
            
            # Insert test hormone cascades
            cascade_names = ["reward_cascade", "stress_response", "learning_enhancement"]
            trigger_hormones = ["dopamine", "cortisol", "serotonin"]
            
            for i in range(10):
                timestamp = (now - timedelta(hours=i)).isoformat()
                cascade_idx = i % len(cascade_names)
                
                affected_hormones = json.dumps(["dopamine", "serotonin"] if cascade_idx != 0 else ["serotonin", "cortisol"])
                affected_lobes = json.dumps(["task_management", "decision_making"] if cascade_idx != 2 else ["memory", "decision_making"])
                effects = json.dumps({"mood_boost": 0.3, "confidence_increase": 0.2})
                feedback_loops = json.dumps(["positive_reinforcement", "stress_reduction"])
                
                cursor.execute("""
                    INSERT INTO hormone_cascades 
                    (timestamp, cascade_name, trigger_hormone, trigger_level,
                     affected_hormones, affected_lobes, duration, effects, feedback_loops)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    cascade_names[cascade_idx],
                    trigger_hormones[cascade_idx],
                    0.7 + 0.1 * (i % 3),
                    affected_hormones,
                    affected_lobes,
                    1.5 + 0.5 * (i % 3),
                    effects,
                    feedback_loops
                ))
            
            # Insert test performance metrics
            components = ["hormone_calculator", "diffusion_engine", "receptor_sensitivity"]
            implementations = ["algorithmic", "neural"]
            metrics = ["accuracy", "latency", "resource_usage"]
            
            for i in range(50):
                timestamp = (now - timedelta(hours=i)).isoformat()
                
                for component in components:
                    for implementation in implementations:
                        for metric in metrics:
                            # Create some patterns in the data
                            if metric == "accuracy":
                                if implementation == "algorithmic":
                                    value = 0.85 + 0.01 * (i % 5)  # Slight oscillation
                                else:  # neural
                                    value = 0.9 + 0.005 * min(i, 20)  # Increasing then plateau
                            elif metric == "latency":
                                if implementation == "algorithmic":
                                    value = 0.15 - 0.001 * i  # Slight improvement
                                else:  # neural
                                    value = 0.2 - 0.005 * min(i, 30)  # Faster improvement then plateau
                            else:  # resource_usage
                                if implementation == "algorithmic":
                                    value = 0.3 + 0.002 * i  # Slight increase
                                else:  # neural
                                    value = 0.5 + 0.005 * min(i, 15)  # Faster increase then plateau
                            
                            cursor.execute("""
                                INSERT INTO performance_metrics 
                                (component, implementation, timestamp, metric_name, metric_value)
                                VALUES (?, ?, ?, ?, ?)
                            """, (component, implementation, timestamp, metric, value))
            
            # Insert test implementation switches
            for i in range(3):
                timestamp = (now - timedelta(days=i*2)).isoformat()
                component = components[i % len(components)]
                
                performance_comparison = json.dumps({
                    "algorithmic": {
                        "accuracy": 0.85,
                        "latency": 0.15,
                        "resource_usage": 0.3
                    },
                    "neural": {
                        "accuracy": 0.95,
                        "latency": 0.1,
                        "resource_usage": 0.5
                    }
                })
                
                expected_improvement = json.dumps({
                    "accuracy": 0.1,
                    "latency": 0.05
                })
                
                cursor.execute("""
                    INSERT INTO implementation_switches 
                    (component, old_implementation, new_implementation, switch_timestamp,
                     reason, performance_comparison, switch_trigger, confidence_score, expected_improvement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    component,
                    "algorithmic",
                    "neural",
                    timestamp,
                    "Neural implementation showed better accuracy",
                    performance_comparison,
                    "performance",
                    0.8,
                    expected_improvement
                ))
            
            conn.commit()
    
    def test_multi_hormone_visualization(self):
        """Test generating multi-hormone visualization data."""
        # Test with default options
        viz_data = self.visualization.generate_multi_hormone_visualization(
            hormone_names=["dopamine", "serotonin", "cortisol"]
        )
        
        # Verify structure and content
        self.assertEqual(viz_data["chart_type"], "line")
        self.assertIn("time_points", viz_data)
        self.assertIn("hormone_data", viz_data)
        self.assertEqual(len(viz_data["hormone_data"]), 3)
        self.assertIn("dopamine", viz_data["hormone_data"])
        self.assertIn("serotonin", viz_data["hormone_data"])
        self.assertIn("cortisol", viz_data["hormone_data"])
        
        # Test with custom options
        options = VisualizationOptions(
            chart_type="area",
            smoothing_factor=0.3,
            include_anomalies=True,
            include_forecast=True
        )
        
        viz_data = self.visualization.generate_multi_hormone_visualization(
            hormone_names=["dopamine", "serotonin"],
            options=options
        )
        
        # Verify options were applied
        self.assertEqual(viz_data["chart_type"], "area")
        self.assertEqual(viz_data["options"]["smoothing_factor"], 0.3)
        self.assertIn("forecasts", viz_data)
        self.assertIn("anomalies", viz_data)
    
    def test_cascade_visualization(self):
        """Test generating cascade visualization data."""
        viz_data = self.visualization.generate_cascade_visualization()
        
        # Verify structure and content
        self.assertIn("cascades", viz_data)
        self.assertIn("network", viz_data)
        self.assertIn("nodes", viz_data["network"])
        self.assertIn("links", viz_data["network"])
        
        # Verify cascade data
        self.assertTrue(len(viz_data["cascades"]) > 0)
        cascade = viz_data["cascades"][0]
        self.assertIn("cascade_name", cascade)
        self.assertIn("trigger_hormone", cascade)
        self.assertIn("affected_hormones", cascade)
        self.assertIn("affected_lobes", cascade)
        
        # Verify network data
        self.assertTrue(len(viz_data["network"]["nodes"]) > 0)
        self.assertTrue(len(viz_data["network"]["links"]) > 0)
        
        # Test with specific cascade name
        viz_data = self.visualization.generate_cascade_visualization(
            cascade_name="reward_cascade"
        )
        
        # Verify filtered results
        self.assertTrue(all(c["cascade_name"] == "reward_cascade" for c in viz_data["cascades"]))
    
    def test_performance_comparison_report(self):
        """Test generating performance comparison report."""
        report = self.visualization.generate_performance_comparison_report("hormone_calculator")
        
        # Verify report structure
        self.assertIsInstance(report, PerformanceComparisonReport)
        self.assertEqual(report.component, "hormone_calculator")
        self.assertIn("algorithmic", report.implementations)
        self.assertIn("neural", report.implementations)
        
        # Verify metrics
        for impl, impl_data in report.implementations.items():
            self.assertIn("metrics", impl_data)
            self.assertIn("time_series", impl_data)
            
            metrics = impl_data["metrics"]
            self.assertIn("accuracy", metrics)
            self.assertIn("latency", metrics)
            self.assertIn("resource_usage", metrics)
        
        # Verify comparison metrics
        self.assertIn("accuracy", report.comparison_metrics)
        self.assertIn("latency", report.comparison_metrics)
        self.assertIn("resource_usage", report.comparison_metrics)
        
        # Verify winner and recommendations
        self.assertIn(report.winner, ["algorithmic", "neural"])
        self.assertTrue(len(report.recommendations) > 0)
        
        # Verify trend analysis
        self.assertIn("algorithmic", report.trend_analysis)
        self.assertIn("neural", report.trend_analysis)
    
    def test_system_performance_report(self):
        """Test generating system performance report."""
        report = self.visualization.generate_system_performance_report()
        
        # Verify report structure
        self.assertIn("component_count", report)
        self.assertIn("system_averages", report)
        self.assertIn("component_reports", report)
        self.assertIn("system_recommendations", report)
        
        # Verify component reports
        self.assertTrue(len(report["component_reports"]) > 0)
        
        # Verify system averages
        averages = report["system_averages"]
        self.assertTrue(any(metric in averages for metric in ["accuracy", "latency", "resource_usage"]))
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        # Test increasing trend
        values = [1.0, 1.1, 1.2, 1.3, 1.4]
        direction, strength = self.visualization._analyze_trend(values)
        self.assertEqual(direction, "increasing")
        self.assertGreater(strength, 0.9)  # Strong trend
        
        # Test decreasing trend
        values = [1.4, 1.3, 1.2, 1.1, 1.0]
        direction, strength = self.visualization._analyze_trend(values)
        self.assertEqual(direction, "decreasing")
        self.assertGreater(strength, 0.9)  # Strong trend
        
        # Test stable trend
        values = [1.0, 1.0, 1.01, 0.99, 1.0]
        direction, strength = self.visualization._analyze_trend(values)
        self.assertEqual(direction, "stable")
        self.assertLess(strength, 0.3)  # Weak trend
        
        # Test fluctuating trend
        values = [1.0, 1.5, 0.8, 1.3, 0.7]
        direction, strength = self.visualization._analyze_trend(values)
        self.assertEqual(direction, "fluctuating")
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        # Test with anomalies
        values = [1.0, 1.1, 1.0, 5.0, 0.9, 1.1]  # 5.0 is an anomaly
        anomalies = self.visualization._detect_value_anomalies(values)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0][0], 3)  # Index of anomaly
        
        # Test without anomalies
        values = [1.0, 1.1, 0.9, 1.0, 1.1, 0.9]
        anomalies = self.visualization._detect_value_anomalies(values)
        self.assertEqual(len(anomalies), 0)
    
    def test_forecast_generation(self):
        """Test forecast generation functionality."""
        # Test with sufficient data
        values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        forecast = self.visualization._generate_forecast(values, 3)
        self.assertEqual(len(forecast), 3)
        
        # Test with insufficient data
        values = [1.0, 1.1]
        forecast = self.visualization._generate_forecast(values, 3)
        self.assertEqual(len(forecast), 0)
    
    def test_smoothing(self):
        """Test smoothing functionality."""
        values = [1.0, 5.0, 1.0, 5.0, 1.0]
        
        # Test with high smoothing factor
        smoothed = self.visualization._apply_smoothing(values, 0.8)
        self.assertEqual(len(smoothed), len(values))
        
        # Verify smoothing effect
        self.assertLess(abs(smoothed[1] - smoothed[3]), abs(values[1] - values[3]))
        
        # Test with low smoothing factor
        smoothed = self.visualization._apply_smoothing(values, 0.2)
        self.assertEqual(len(smoothed), len(values))
        
        # Verify stronger smoothing effect
        self.assertLess(abs(smoothed[1] - smoothed[3]), abs(values[1] - values[3]))


class TestMonitoringSystemWithVisualization(unittest.TestCase):
    """Test suite for integrated monitoring system with visualization."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary database for testing
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self.monitoring = MonitoringSystemWithVisualization(self.temp_db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        self.monitoring.stop_monitoring()
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)
    
    def test_record_performance_metrics(self):
        """Test recording performance metrics."""
        # Record metrics
        self.monitoring.record_performance_metrics(
            component="test_component",
            implementation="test_impl",
            metrics={
                "accuracy": 0.95,
                "latency": 0.1
            }
        )
        
        # Verify metrics were recorded
        with sqlite3.connect(self.temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM performance_metrics
                WHERE component = ? AND implementation = ?
            """, ("test_component", "test_impl"))
            
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)  # Two metrics recorded
    
    def test_record_implementation_switch(self):
        """Test recording implementation switch."""
        # Record switch
        self.monitoring.record_implementation_switch(
            component="test_component",
            old_implementation="old_impl",
            new_implementation="new_impl",
            reason="Test reason",
            performance_comparison={
                "old_impl": {"accuracy": 0.8},
                "new_impl": {"accuracy": 0.9}
            }
        )
        
        # Verify switch was recorded
        with sqlite3.connect(self.temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM implementation_switches
                WHERE component = ? AND old_implementation = ? AND new_implementation = ?
            """, ("test_component", "old_impl", "new_impl"))
            
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()