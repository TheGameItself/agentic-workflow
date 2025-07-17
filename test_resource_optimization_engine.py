"""
Test suite for ResourceOptimizationEngine.

This module contains tests for the ResourceOptimizationEngine class, which
dynamically optimizes resource usage based on workload patterns and system constraints.
"""

import unittest
import asyncio
from unittest.mock import MagicMock, patch
import time
from datetime import datetime, timedelta

from src.mcp.resource_optimization_engine import (
    ResourceOptimizationEngine,
    ResourceMetrics,
    ResourceConstraints,
    WorkloadPattern,
    ResourcePrediction,
    TrainingSchedule,
    AdaptationPlan,
    RecoveryState
)


class TestResourceOptimizationEngine(unittest.TestCase):
    """Test cases for ResourceOptimizationEngine."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock dependencies
        self.mock_hormone_controller = MagicMock()
        self.mock_brain_state_aggregator = MagicMock()
        self.mock_event_bus = MagicMock()
        
        # Configure mock hormone controller
        self.mock_hormone_controller.get_hormone_levels.return_value = {
            "dopamine": 0.6,
            "serotonin": 0.7,
            "cortisol": 0.3,
            "adrenaline": 0.2,
            "oxytocin": 0.5,
            "vasopressin": 0.4,
            "growth_hormone": 0.5,
            "norepinephrine": 0.3
        }
        
        # Create engine with mocks
        self.engine = ResourceOptimizationEngine(
            hormone_controller=self.mock_hormone_controller,
            brain_state_aggregator=self.mock_brain_state_aggregator,
            event_bus=self.mock_event_bus
        )
        
        # Patch psutil to avoid actual system calls
        self.psutil_patcher = patch('src.mcp.resource_optimization_engine.psutil')
        self.mock_psutil = self.psutil_patcher.start()
        
        # Configure mock psutil
        self.mock_psutil.cpu_percent.return_value = 30.0
        mock_memory = MagicMock()
        mock_memory.percent = 40.0
        mock_memory.available = 8000000000  # 8GB
        self.mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        self.mock_psutil.disk_usage.return_value = mock_disk
        
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000
        self.mock_psutil.net_io_counters.return_value = mock_net_io
        
        mock_cpu_times = MagicMock()
        mock_cpu_times.iowait = 2.0
        self.mock_psutil.cpu_times_percent.return_value = mock_cpu_times
    
    def tearDown(self):
        """Clean up after tests."""
        self.psutil_patcher.stop()
    
    def test_initialization(self):
        """Test initialization of ResourceOptimizationEngine."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.hormone_controller, self.mock_hormone_controller)
        self.assertEqual(self.engine.brain_state_aggregator, self.mock_brain_state_aggregator)
        self.assertEqual(self.engine.event_bus, self.mock_event_bus)
        self.assertFalse(self.engine.hormone_adjustment_active)
        self.assertFalse(self.engine.memory_consolidation_active)
        self.assertFalse(self.engine.in_recovery_mode)
    
    def test_collect_resource_metrics(self):
        """Test collection of resource metrics."""
        metrics = self.engine._collect_resource_metrics()
        
        self.assertEqual(metrics.cpu_usage, 30.0)
        self.assertEqual(metrics.memory_usage, 40.0)
        self.assertEqual(metrics.memory_available, 8000000000)
        self.assertEqual(metrics.disk_usage, 50.0)
        self.assertEqual(metrics.network_usage, 3000)
        self.assertEqual(metrics.io_wait, 2.0)
        self.assertIsNotNone(metrics.timestamp)
    
    def test_adjust_hormone_production_rates(self):
        """Test adjustment of hormone production rates."""
        # Call the method
        self.engine.adjust_hormone_production_rates(0.8)
        
        # Check that hormone levels were retrieved
        self.mock_hormone_controller.get_hormone_levels.assert_called_once()
        
        # Check that hormone adjustment is active
        self.assertTrue(self.engine.hormone_adjustment_active)
        
        # Check that hormone levels were adjusted
        self.assertGreater(len(self.engine.hormone_adjustment_factors), 0)
        
        # Check that event was emitted
        self.mock_event_bus.emit.assert_called_with(
            "hormone_production_adjusted",
            {
                "computational_load": 0.8,
                "adjustment_factors": self.engine.hormone_adjustment_factors
            }
        )
    
    def test_trigger_memory_consolidation(self):
        """Test triggering of memory consolidation."""
        # Call the method
        self.engine.trigger_memory_consolidation(0.85, 0.8)
        
        # Check that memory consolidation is active
        self.assertTrue(self.engine.memory_consolidation_active)
        
        # Check that vasopressin was released
        self.mock_hormone_controller.release_hormone.assert_called_with(
            "memory",
            "vasopressin",
            0.9,
            context={"event_type": "memory_consolidation", "memory_usage": 0.85}
        )
        
        # Check that event was emitted
        self.mock_event_bus.emit.assert_called_with(
            "memory_consolidation_triggered",
            {
                "memory_usage": 0.85,
                "threshold": 0.8,
                "timestamp": unittest.mock.ANY
            }
        )
    
    def test_schedule_background_training(self):
        """Test scheduling of background training."""
        # Create resource metrics
        resources = ResourceMetrics(cpu_usage=15.0, memory_usage=40.0)
        
        # Call the method
        schedule = self.engine.schedule_background_training(resources)
        
        # Check that schedule was created
        self.assertIsNotNone(schedule)
        self.assertGreater(len(schedule.tasks), 0)
        
        # Check that active training tasks were updated
        self.assertGreater(len(self.engine.active_training_tasks), 0)
        
        # Check that event was emitted
        self.mock_event_bus.emit.assert_called_with(
            "background_training_scheduled",
            {
                "schedule": {
                    "tasks": len(schedule.tasks),
                    "resource_allocation": schedule.resource_allocation
                }
            }
        )
    
    def test_prioritize_lobe_resources(self):
        """Test prioritization of lobe resources."""
        # Define active lobes
        active_lobes = ["memory", "pattern_recognition"]
        
        # Define current allocation
        current_allocation = {
            "memory": 0.2,
            "pattern_recognition": 0.2,
            "task_management": 0.2,
            "decision_making": 0.2
        }
        
        # Call the method
        updated_allocation = self.engine.prioritize_lobe_resources(active_lobes, current_allocation)
        
        # Check that allocation was updated
        self.assertIsNotNone(updated_allocation)
        self.assertEqual(len(updated_allocation), 4)
        
        # Check that active lobes got higher allocation
        self.assertGreater(updated_allocation["memory"], updated_allocation["task_management"])
        self.assertGreater(updated_allocation["pattern_recognition"], updated_allocation["decision_making"])
        
        # Check that norepinephrine was released for active lobes
        self.assertEqual(self.mock_hormone_controller.release_hormone.call_count, 2)
    
    def test_adapt_to_resource_constraints(self):
        """Test adaptation to resource constraints."""
        # Set up high resource usage
        self.mock_psutil.cpu_percent.return_value = 85.0
        mock_memory = MagicMock()
        mock_memory.percent = 90.0
        self.mock_psutil.virtual_memory.return_value = mock_memory
        
        # Update metrics
        self.engine.current_metrics = self.engine._collect_resource_metrics()
        
        # Define constraints
        constraints = ResourceConstraints(
            max_cpu_usage=70.0,
            max_memory_usage=75.0,
            priority_lobes=["task_management"]
        )
        
        # Call the method
        plan = self.engine.adapt_to_resource_constraints(constraints)
        
        # Check that plan was created
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.hormone_adjustments), 0)
        self.assertGreater(len(plan.lobe_priority_changes), 0)
        self.assertGreater(len(plan.memory_consolidation_targets), 0)
        
        # Check that hormone levels were adjusted
        self.assertEqual(self.mock_hormone_controller.set_hormone_level.call_count, 
                        len(plan.hormone_adjustments))
        
        # Check that event was emitted
        self.mock_event_bus.emit.assert_called_with(
            "resource_adaptation_plan_created",
            unittest.mock.ANY
        )
    
    def test_restore_optimal_levels(self):
        """Test restoration of optimal hormone levels."""
        # Define recovery state
        recovery_state = RecoveryState(
            previous_hormone_levels={
                "dopamine": 0.3,
                "serotonin": 0.4,
                "cortisol": 0.8
            },
            optimal_hormone_levels={
                "dopamine": 0.6,
                "serotonin": 0.7,
                "cortisol": 0.3
            },
            recovery_duration=60.0
        )
        
        # Call the method
        self.engine.restore_optimal_levels(recovery_state)
        
        # Check that recovery mode is active
        self.assertTrue(self.engine.in_recovery_mode)
        self.assertIsNotNone(self.engine.recovery_state)
        
        # Check that event was emitted
        self.mock_event_bus.emit.assert_called_with(
            "resource_recovery_started",
            {
                "recovery_duration": 60.0,
                "timestamp": unittest.mock.ANY
            }
        )


if __name__ == "__main__":
    unittest.main()