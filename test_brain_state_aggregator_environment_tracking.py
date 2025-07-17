#!/usr/bin/env python3
"""
Unit tests for Brain State Aggregator environment state tracking functionality.
Tests the integration of environment state collection from various system components.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mcp.brain_state_aggregator import BrainStateAggregator


class TestBrainStateAggregatorEnvironmentTracking(unittest.TestCase):
    """Test environment state tracking functionality in Brain State Aggregator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_lobes = {
            "memory_lobe": Mock(),
            "decision_lobe": Mock(),
            "pattern_lobe": Mock()
        }
        
        self.mock_hormone_engine = Mock()
        self.mock_sensory_column = Mock()
        self.mock_vector_memory = Mock()
        self.mock_event_bus = Mock()
        
        # Configure mock methods
        self.mock_lobes["memory_lobe"].get_state.return_value = {"memory_usage": 0.7, "active": True}
        self.mock_lobes["decision_lobe"].get_state.return_value = {"decisions_made": 15, "confidence": 0.8}
        self.mock_lobes["pattern_lobe"].get_state.return_value = {"patterns_detected": 5, "accuracy": 0.9}
        
        self.mock_hormone_engine.get_levels.return_value = {
            "dopamine": 0.6,
            "serotonin": 0.7,
            "cortisol": 0.3
        }
        
        self.mock_sensory_column.get_latest.return_value = {
            "input_type": "text",
            "processing_time": 0.05,
            "confidence": 0.85
        }
        
        self.mock_vector_memory.get_relevant_vectors.return_value = [
            {"id": 1, "similarity": 0.9},
            {"id": 2, "similarity": 0.8}
        ]
        
        # Create Brain State Aggregator instance
        self.bsa = BrainStateAggregator(
            lobes=self.mock_lobes,
            hormone_engine=self.mock_hormone_engine,
            sensory_column=self.mock_sensory_column,
            vector_memory=self.mock_vector_memory,
            event_bus=self.mock_event_bus
        )
        
        # Add some test data
        self.bsa.update_hormone_levels({"dopamine": 0.6, "serotonin": 0.7}, "memory_lobe")
        self.bsa.register_implementation_performance("hormone_calculator", "neural", {
            "accuracy": 0.85,
            "latency": 0.02,
            "resource_usage": 0.3
        })
    
    def test_get_environment_state_basic(self):
        """Test basic environment state collection."""
        env_state = self.bsa.get_environment_state()
        
        # Check that all required sections are present
        self.assertIn("timestamp", env_state)
        self.assertIn("system_state", env_state)
        self.assertIn("lobe_states", env_state)
        self.assertIn("hormone_state", env_state)
        self.assertIn("performance_state", env_state)
        self.assertIn("resource_state", env_state)
        self.assertIn("sensory_state", env_state)
        self.assertIn("memory_state", env_state)
        
        # Check timestamp format
        self.assertIsInstance(env_state["timestamp"], str)
        datetime.fromisoformat(env_state["timestamp"])  # Should not raise exception
    
    def test_collect_system_state(self):
        """Test system state collection."""
        env_state = self.bsa.get_environment_state()
        system_state = env_state["system_state"]
        
        # Check required system state fields
        self.assertIn("active_implementations", system_state)
        self.assertIn("buffer_count", system_state)
        self.assertIn("event_bus_active", system_state)
        
        # Verify values
        self.assertEqual(system_state["buffer_count"], len(self.bsa.buffers))
        self.assertTrue(system_state["event_bus_active"])
        self.assertIsInstance(system_state["active_implementations"], dict)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_system_state_with_psutil(self, mock_disk, mock_memory, mock_cpu):
        """Test system state collection with psutil available."""
        # Mock psutil functions
        mock_cpu.return_value = 45.2
        mock_memory.return_value = Mock(percent=67.8)
        mock_disk.return_value = Mock(percent=23.4)
        
        env_state = self.bsa.get_environment_state()
        system_state = env_state["system_state"]
        
        # Check that psutil data is included
        self.assertEqual(system_state["cpu_usage"], 45.2)
        self.assertEqual(system_state["memory_usage"], 67.8)
        self.assertEqual(system_state["disk_usage"], 23.4)
    
    def test_collect_lobe_states(self):
        """Test lobe state collection."""
        env_state = self.bsa.get_environment_state()
        lobe_states = env_state["lobe_states"]
        
        # Check that all lobes are represented
        self.assertEqual(len(lobe_states), len(self.mock_lobes))
        
        for lobe_name in self.mock_lobes.keys():
            self.assertIn(lobe_name, lobe_states)
            lobe_state = lobe_states[lobe_name]
            
            # Check required fields
            self.assertIn("name", lobe_state)
            self.assertIn("active", lobe_state)
            self.assertIn("state_data", lobe_state)
            self.assertIn("hormone_production", lobe_state)
            self.assertIn("in_buffer", lobe_state)
            self.assertIn("access_count", lobe_state)
            
            # Verify values
            self.assertEqual(lobe_state["name"], lobe_name)
            self.assertTrue(lobe_state["active"])
            self.assertIsNotNone(lobe_state["state_data"])
    
    def test_collect_hormone_state(self):
        """Test hormone state collection."""
        env_state = self.bsa.get_environment_state()
        hormone_state = env_state["hormone_state"]
        
        # Check required fields
        self.assertIn("current_levels", hormone_state)
        self.assertIn("recent_changes", hormone_state)
        self.assertIn("cascade_activity", hormone_state)
        self.assertIn("threshold_violations", hormone_state)
        self.assertIn("trend_analysis", hormone_state)
        self.assertIn("source_distribution", hormone_state)
        self.assertIn("total_hormones", hormone_state)
        
        # Verify hormone levels are included
        self.assertIn("dopamine", hormone_state["current_levels"])
        self.assertIn("serotonin", hormone_state["current_levels"])
        
        # Check that source tracking is working
        self.assertIn("dopamine", hormone_state["source_distribution"])
        self.assertIn("memory_lobe", hormone_state["source_distribution"]["dopamine"])
    
    def test_collect_performance_state(self):
        """Test performance state collection."""
        env_state = self.bsa.get_environment_state()
        performance_state = env_state["performance_state"]
        
        # Check required fields
        self.assertIn("active_implementations", performance_state)
        self.assertIn("recent_switches", performance_state)
        self.assertIn("performance_summary", performance_state)
        self.assertIn("neural_availability", performance_state)
        self.assertIn("fallback_status", performance_state)
        
        # Verify performance data is included
        self.assertIn("hormone_calculator", performance_state["active_implementations"])
        self.assertIn("hormone_calculator", performance_state["performance_summary"])
    
    def test_collect_resource_state(self):
        """Test resource state collection."""
        env_state = self.bsa.get_environment_state()
        resource_state = env_state["resource_state"]
        
        # Check required fields
        self.assertIn("memory_usage", resource_state)
        self.assertIn("computational_load", resource_state)
        self.assertIn("buffer_efficiency", resource_state)
        
        # Verify memory usage tracking
        memory_usage = resource_state["memory_usage"]
        self.assertIn("hormone_history_size", memory_usage)
        self.assertIn("performance_history_size", memory_usage)
        self.assertIn("buffer_size", memory_usage)
        
        # Verify computational load tracking
        comp_load = resource_state["computational_load"]
        self.assertIn("active_lobes", comp_load)
        self.assertIn("hormone_calculations", comp_load)
        self.assertIn("performance_comparisons", comp_load)
    
    def test_collect_sensory_state(self):
        """Test sensory state collection."""
        env_state = self.bsa.get_environment_state()
        sensory_state = env_state["sensory_state"]
        
        # Check required fields
        self.assertIn("sensory_column_active", sensory_state)
        self.assertIn("latest_sensory_data", sensory_state)
        self.assertIn("sensory_processing_status", sensory_state)
        
        # Verify sensory data is collected
        self.assertTrue(sensory_state["sensory_column_active"])
        self.assertEqual(sensory_state["sensory_processing_status"], "active")
        self.assertIsNotNone(sensory_state["latest_sensory_data"])
    
    def test_collect_memory_state(self):
        """Test memory state collection."""
        env_state = self.bsa.get_environment_state()
        memory_state = env_state["memory_state"]
        
        # Check required fields
        self.assertIn("vector_memory_active", memory_state)
        self.assertIn("vector_count", memory_state)
        
        # Verify memory data is collected
        self.assertTrue(memory_state["vector_memory_active"])
        self.assertEqual(memory_state["vector_count"], 2)  # Based on mock data
    
    def test_environment_state_in_context_package(self):
        """Test that environment state is included in context packages."""
        context_package = self.bsa.get_context_package("memory_lobe")
        
        # Check that environment state is included
        self.assertIn("environment", context_package)
        
        # Verify structure is correct (timestamps will differ due to timing)
        env_state = self.bsa.get_environment_state()
        self.assertEqual(len(context_package["environment"]["lobe_states"]), 
                        len(env_state["lobe_states"]))
        
        # Check that both have the same structure
        for key in ["system_state", "lobe_states", "hormone_state", "performance_state"]:
            self.assertIn(key, context_package["environment"])
            self.assertIn(key, env_state)
    
    def test_environment_state_error_handling(self):
        """Test error handling in environment state collection."""
        # Create a BSA with a problematic lobe
        problematic_lobe = Mock()
        problematic_lobe.get_state.side_effect = Exception("Test error")
        
        lobes_with_error = self.mock_lobes.copy()
        lobes_with_error["problematic_lobe"] = problematic_lobe
        
        bsa_with_error = BrainStateAggregator(lobes=lobes_with_error)
        
        # Should not raise exception, but should handle error gracefully
        env_state = bsa_with_error.get_environment_state()
        
        # Check that error is handled
        self.assertIn("lobe_states", env_state)
        self.assertIn("problematic_lobe", env_state["lobe_states"])
        
        problematic_state = env_state["lobe_states"]["problematic_lobe"]
        self.assertFalse(problematic_state["active"])
        self.assertIn("error", problematic_state)
    
    def test_environment_state_without_components(self):
        """Test environment state collection when components are None."""
        # Create BSA without optional components
        minimal_bsa = BrainStateAggregator()
        
        env_state = minimal_bsa.get_environment_state()
        
        # Should still return valid structure
        self.assertIn("timestamp", env_state)
        self.assertIn("system_state", env_state)
        self.assertIn("lobe_states", env_state)
        self.assertIn("hormone_state", env_state)
        
        # Check that missing components are handled gracefully
        self.assertFalse(env_state["sensory_state"]["sensory_column_active"])
        self.assertFalse(env_state["memory_state"]["vector_memory_active"])
        self.assertEqual(env_state["sensory_state"]["sensory_processing_status"], "inactive")
    
    def test_hormone_threshold_violations(self):
        """Test detection of hormone threshold violations."""
        # Set some thresholds
        self.bsa.set_hormone_threshold("dopamine", "activation", 0.5)
        self.bsa.set_hormone_threshold("cortisol", "stress", 0.4)
        
        # Update hormone levels to trigger violations
        self.bsa.update_hormone_levels({"dopamine": 0.8, "cortisol": 0.6})
        
        env_state = self.bsa.get_environment_state()
        hormone_state = env_state["hormone_state"]
        
        # Check for threshold violations
        violations = hormone_state["threshold_violations"]
        self.assertGreater(len(violations), 0)
        
        # Find dopamine violation
        dopamine_violation = next((v for v in violations if v["hormone"] == "dopamine"), None)
        self.assertIsNotNone(dopamine_violation)
        self.assertEqual(dopamine_violation["threshold_type"], "activation")
        self.assertEqual(dopamine_violation["current_level"], 0.8)
        self.assertEqual(dopamine_violation["threshold_value"], 0.5)
        self.assertAlmostEqual(dopamine_violation["excess"], 0.3, places=5)
    
    def test_performance_comparison_due_tracking(self):
        """Test tracking of when performance comparisons are due."""
        # Set comparison frequency
        self.bsa.set_comparison_frequency("hormone_calculator", 30)  # 30 seconds
        
        env_state = self.bsa.get_environment_state()
        performance_state = env_state["performance_state"]
        
        # Check comparison due tracking
        self.assertIn("comparison_due", performance_state)
        self.assertIn("hormone_calculator", performance_state["comparison_due"])
        
        comparison_info = performance_state["comparison_due"]["hormone_calculator"]
        self.assertIn("seconds_since_last", comparison_info)
        self.assertIn("frequency", comparison_info)
        self.assertIn("due", comparison_info)
        self.assertEqual(comparison_info["frequency"], 30)
    
    def test_buffer_efficiency_metrics(self):
        """Test buffer efficiency metrics calculation."""
        # Simulate some buffer accesses
        self.bsa.prefetch_history = ["memory_lobe", "decision_lobe", "memory_lobe", "pattern_lobe"]
        
        env_state = self.bsa.get_environment_state()
        resource_state = env_state["resource_state"]
        
        # Check buffer efficiency metrics
        self.assertIn("buffer_efficiency", resource_state)
        efficiency = resource_state["buffer_efficiency"]
        
        self.assertIn("hit_ratio", efficiency)
        self.assertIn("total_accesses", efficiency)
        self.assertIn("unique_accesses", efficiency)
        
        self.assertEqual(efficiency["total_accesses"], 4)
        self.assertEqual(efficiency["unique_accesses"], 3)  # memory_lobe, decision_lobe, pattern_lobe
        self.assertEqual(efficiency["hit_ratio"], 3/4)  # 3 unique out of 4 total
    
    def test_volatility_calculation(self):
        """Test hormone volatility calculation."""
        # Create some hormone history with varying levels
        test_entries = [
            {"level": 0.5}, {"level": 0.7}, {"level": 0.4}, {"level": 0.8}, {"level": 0.6}
        ]
        
        volatility = self.bsa._calculate_volatility(test_entries)
        
        # Should return a positive volatility value
        self.assertGreater(volatility, 0)
        self.assertIsInstance(volatility, float)
        
        # Test with stable levels
        stable_entries = [{"level": 0.5}, {"level": 0.5}, {"level": 0.5}]
        stable_volatility = self.bsa._calculate_volatility(stable_entries)
        
        # Should be very low volatility
        self.assertAlmostEqual(stable_volatility, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()