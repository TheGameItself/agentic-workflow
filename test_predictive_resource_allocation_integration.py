"""
Test script for predictive resource allocation integration.

This script tests the integration of PredictiveResourceAllocation with ResourceOptimizationEngine.
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from src.mcp.predictive_resource_allocation import (
    PredictiveResourceAllocation,
    ResourceMetrics,
    ResourceConstraints,
    AdaptationPlan
)
from src.mcp.predictive_resource_allocation_integration import PredictiveResourceAllocationIntegration
from src.mcp.workload_pattern_analyzer import WorkloadPattern, ResourcePrediction


class MockResourceOptimizationEngine:
    """Mock resource optimization engine for testing."""
    
    def __init__(self):
        self.hormone_levels = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.3,
            "adrenaline": 0.2,
            "growth_hormone": 0.6,
            "vasopressin": 0.4
        }
        self.memory_consolidation_triggered = False
        self.background_training_scheduled = False
        self.adaptation_plans = []
        self.active_lobes = []
        self.resource_allocation = {}
    
    def trigger_memory_consolidation(self, memory_usage: float, threshold: float):
        """Trigger memory consolidation."""
        self.memory_consolidation_triggered = True
        print(f"Memory consolidation triggered: {memory_usage:.2f} (threshold: {threshold:.2f})")
        return True
    
    def schedule_background_training(self, available_resources: ResourceMetrics):
        """Schedule background training."""
        self.background_training_scheduled = True
        print(f"Background training scheduled: CPU {available_resources.cpu_usage:.1f}% available")
        return {}
    
    def prioritize_lobe_resources(self, active_lobes: List[str], resource_allocation: Dict[str, float]):
        """Prioritize resources for active lobes."""
        self.active_lobes = active_lobes
        self.resource_allocation = resource_allocation.copy() if resource_allocation else {}
        
        # Simple allocation: equal distribution
        result = {}
        if active_lobes:
            per_lobe = 100.0 / len(active_lobes)
            for lobe in active_lobes:
                result[lobe] = per_lobe
        
        print(f"Resources prioritized for lobes: {', '.join(active_lobes)}")
        return result
    
    def predict_workload_pattern(self):
        """Predict workload pattern."""
        pattern = WorkloadPattern(
            pattern_type="cyclic",
            periodicity=60.0,
            confidence=0.8
        )
        print(f"Predicted workload pattern: {pattern.pattern_type}")
        return pattern
    
    def predict_resource_needs(self, pattern):
        """Predict resource needs."""
        prediction = ResourcePrediction(
            confidence=0.7,
            prediction_horizon=60
        )
        
        # Add some predicted values
        for i in range(60):
            time_point = (datetime.now() + timedelta(seconds=i)).strftime("%H:%M:%S")
            prediction.predicted_cpu[time_point] = 50.0 + 20.0 * abs(((i % 60) / 30.0) - 1.0)
            prediction.predicted_memory[time_point] = 60.0 + 10.0 * abs(((i % 60) / 30.0) - 1.0)
        
        print(f"Predicted resource needs: {prediction.confidence:.2f} confidence")
        return prediction
    
    def _apply_predictive_optimizations(self, prediction):
        """Apply predictive optimizations."""
        print(f"Applied predictive optimizations based on prediction with {prediction.confidence:.2f} confidence")
        return True
    
    def _apply_adaptation_plan(self, plan: AdaptationPlan):
        """Apply adaptation plan."""
        self.adaptation_plans.append(plan)
        print(f"Applied adaptation plan with {len(plan.hormone_adjustments)} hormone adjustments")
        return True
    
    def restore_optimal_levels(self, recovery_state):
        """Restore optimal hormone levels."""
        print(f"Restoring optimal hormone levels for {len(recovery_state.optimal_hormone_levels)} hormones")
        return True


class MockHormoneController:
    """Mock hormone controller for testing."""
    
    def __init__(self):
        self.hormone_levels = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "cortisol": 0.3,
            "adrenaline": 0.2,
            "growth_hormone": 0.6,
            "vasopressin": 0.4
        }
    
    def get_hormone_levels(self) -> Dict[str, float]:
        """Get current hormone levels."""
        return self.hormone_levels.copy()
    
    def set_hormone_level(self, hormone: str, level: float):
        """Set hormone level."""
        self.hormone_levels[hormone] = level
        print(f"Set {hormone} to {level:.2f}")


class MockBrainStateAggregator:
    """Mock brain state aggregator for testing."""
    
    def __init__(self):
        self.environment_state = {}
    
    def get_environment_state(self) -> Dict[str, Any]:
        """Get environment state."""
        return self.environment_state
    
    def update_environment_state(self, state: Dict[str, Any]):
        """Update environment state."""
        self.environment_state.update(state)
        print(f"Updated environment state with {len(state)} keys")


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.events = []
        self.subscriptions = {}
    
    def emit(self, event_type: str, data: Any):
        """Emit an event."""
        self.events.append((event_type, data))
        print(f"Event emitted: {event_type}")
        
        # Call subscribers
        if event_type in self.subscriptions:
            for callback in self.subscriptions[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in event handler: {e}")
    
    def subscribe(self, event_type: str, callback):
        """Subscribe to an event."""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(callback)
        print(f"Subscribed to event: {event_type}")


def test_predictive_resource_allocation_integration():
    """Test the PredictiveResourceAllocationIntegration class."""
    print("\n=== Testing PredictiveResourceAllocationIntegration ===")
    
    # Create mock components
    resource_optimization_engine = MockResourceOptimizationEngine()
    hormone_controller = MockHormoneController()
    brain_state_aggregator = MockBrainStateAggregator()
    event_bus = MockEventBus()
    
    # Create integration
    integration = PredictiveResourceAllocationIntegration(
        resource_optimization_engine=resource_optimization_engine,
        brain_state_aggregator=brain_state_aggregator,
        event_bus=event_bus,
        hormone_controller=hormone_controller
    )
    
    # Start integration
    integration.start()
    
    try:
        # Wait for initialization
        print("\nWaiting for initialization...")
        time.sleep(1)
        
        # Test resource constraint event
        print("\nTesting resource constraint event...")
        event_bus.emit("resource_constraint_detected", {
            "constraint_type": "cpu",
            "severity": 0.7,
            "max_cpu_usage": 70.0,
            "max_memory_usage": 80.0,
            "max_disk_usage": 90.0,
            "priority_lobes": ["memory", "task_management"]
        })
        
        # Wait for event processing
        time.sleep(1)
        
        # Test memory pressure event
        print("\nTesting memory pressure event...")
        event_bus.emit("memory_pressure_detected", {
            "memory_usage": 0.8,
            "threshold": 0.7
        })
        
        # Wait for event processing
        time.sleep(1)
        
        # Test system idle event
        print("\nTesting system idle event...")
        event_bus.emit("system_idle_detected", {
            "cpu_usage": 20.0,
            "memory_usage": 40.0,
            "memory_available": 1000000000,
            "disk_usage": 50.0
        })
        
        # Wait for event processing
        time.sleep(1)
        
        # Test high activity event
        print("\nTesting high activity event...")
        event_bus.emit("high_activity_detected", {
            "active_lobes": ["memory", "pattern_recognition", "task_management"],
            "resource_allocation": {
                "memory": 30.0,
                "pattern_recognition": 20.0,
                "task_management": 25.0
            }
        })
        
        # Wait for event processing
        time.sleep(1)
        
        # Test resource recovery event
        print("\nTesting resource recovery event...")
        event_bus.emit("resource_recovery_detected", {
            "previous_hormone_levels": {
                "dopamine": 0.3,
                "serotonin": 0.4,
                "cortisol": 0.7
            },
            "optimal_hormone_levels": {
                "dopamine": 0.6,
                "serotonin": 0.7,
                "cortisol": 0.3
            }
        })
        
        # Wait for event processing
        time.sleep(1)
        
        # Check current prediction
        current_prediction = integration.get_current_prediction()
        if current_prediction:
            print(f"\nCurrent prediction: {current_prediction.confidence:.2f} confidence")
        else:
            print("\nNo current prediction available")
        
        # Check current adaptation plan
        current_plan = integration.get_current_adaptation_plan()
        if current_plan:
            print(f"Current adaptation plan: {len(current_plan.hormone_adjustments)} hormone adjustments")
        else:
            print("No current adaptation plan available")
        
        # Check recovery mode
        recovery_mode = integration.is_in_recovery_mode()
        recovery_progress = integration.get_recovery_progress()
        print(f"Recovery mode: {recovery_mode}, Progress: {recovery_progress:.2f}")
        
    finally:
        # Stop integration
        integration.stop()
        print("\nIntegration stopped")


if __name__ == "__main__":
    # Test predictive resource allocation integration
    test_predictive_resource_allocation_integration()