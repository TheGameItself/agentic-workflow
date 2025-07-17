"""
Simple test script for predictive resource allocation and constraint adaptation.

This script tests the functionality of the PredictiveResourceAllocation class
without requiring external dependencies like numpy or matplotlib.
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from src.mcp.workload_pattern_analyzer import WorkloadPatternAnalyzer, WorkloadPattern, ResourcePrediction
from src.mcp.predictive_resource_allocation import PredictiveResourceAllocation, ResourceMetrics, ResourceConstraints, AdaptationPlan


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


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.events = []
    
    def emit(self, event_type: str, data: Any):
        """Emit an event."""
        self.events.append((event_type, data))
        print(f"Event: {event_type}")


def generate_simple_data(num_points: int = 100) -> List[float]:
    """Generate simple test data without numpy."""
    data = []
    for i in range(num_points):
        # Simple sine wave with some randomness
        value = 50.0 + 30.0 * (i % 20) / 20.0
        value += random.uniform(-5.0, 5.0)
        data.append(min(100.0, max(0.0, value)))
    return data


def test_predictive_resource_allocation():
    """Test the PredictiveResourceAllocation class."""
    print("\n=== Testing PredictiveResourceAllocation ===")
    
    # Create mock components
    hormone_controller = MockHormoneController()
    event_bus = MockEventBus()
    
    # Create resource allocation system
    resource_allocator = PredictiveResourceAllocation(
        hormone_controller=hormone_controller,
        event_bus=event_bus
    )
    
    # Generate some metrics
    for i in range(100):
        # Simulate increasing CPU usage
        cpu_usage = 40.0 + i * 0.5
        memory_usage = 50.0 + i * 0.3
        disk_usage = 60.0 + i * 0.1
        
        # Update metrics
        metrics = ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage
        )
        resource_allocator.current_metrics = metrics
        resource_allocator.metrics_history.append(metrics)
        
        # Update every 10 iterations
        if i % 10 == 0:
            resource_allocator.update()
    
    # Test constraint adaptation
    print("\nTesting constraint adaptation...")
    constraints = ResourceConstraints(
        max_cpu_usage=70.0,
        max_memory_usage=70.0,
        max_disk_usage=80.0,
        priority_lobes=["memory", "task_management"]
    )
    
    adaptation_plan = resource_allocator.adapt_to_resource_constraints(constraints)
    
    print(f"Hormone adjustments: {len(adaptation_plan.hormone_adjustments)}")
    for hormone, level in adaptation_plan.hormone_adjustments.items():
        print(f"  {hormone}: {level:.2f}")
    
    print(f"Lobe priority changes: {len(adaptation_plan.lobe_priority_changes)}")
    for lobe, priority in adaptation_plan.lobe_priority_changes.items():
        print(f"  {lobe}: {'+' if priority > 0 else ''}{priority}")
    
    print(f"Memory consolidation targets: {adaptation_plan.memory_consolidation_targets}")
    print(f"Background task adjustments: {len(adaptation_plan.background_task_adjustments)}")
    print(f"Estimated resource savings: {adaptation_plan.estimated_resource_savings}")
    
    # Apply adaptation plan
    resource_allocator._apply_adaptation_plan(adaptation_plan)
    
    # Check hormone levels after adaptation
    print("\nHormone levels after adaptation:")
    for hormone, level in hormone_controller.hormone_levels.items():
        print(f"  {hormone}: {level:.2f}")
    
    # Check events emitted
    print("\nEvents emitted:")
    for event_type, _ in event_bus.events:
        print(f"  {event_type}")


if __name__ == "__main__":
    # Test predictive resource allocation
    test_predictive_resource_allocation()