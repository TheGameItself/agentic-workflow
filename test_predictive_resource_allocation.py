"""
Test script for predictive resource allocation and constraint adaptation.

This script tests the functionality of the WorkloadPatternAnalyzer and
PredictiveResourceAllocation classes.
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Try to import numpy and matplotlib, but continue if not available
try:
    import numpy as np
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: numpy or matplotlib not available. Plotting will be disabled.")
    PLOTTING_AVAILABLE = False

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


def generate_cyclic_workload(num_points: int = 500, period: int = 100, 
                           amplitude: float = 30.0, base: float = 40.0, 
                           noise: float = 5.0) -> List[float]:
    """Generate cyclic workload pattern with noise."""
    x = np.arange(num_points)
    y = base + amplitude * np.sin(2 * np.pi * x / period)
    y += np.random.normal(0, noise, num_points)
    return np.clip(y, 0, 100).tolist()


def generate_bursty_workload(num_points: int = 500, base: float = 20.0,
                           burst_height: float = 60.0, burst_freq: float = 0.05,
                           noise: float = 3.0) -> List[float]:
    """Generate bursty workload pattern with noise."""
    y = np.ones(num_points) * base
    for i in range(num_points):
        if random.random() < burst_freq:
            burst_length = random.randint(5, 15)
            burst_start = i
            burst_end = min(i + burst_length, num_points)
            y[burst_start:burst_end] = base + burst_height
    y += np.random.normal(0, noise, num_points)
    return np.clip(y, 0, 100).tolist()


def test_workload_pattern_analyzer():
    """Test the WorkloadPatternAnalyzer class."""
    print("\n=== Testing WorkloadPatternAnalyzer ===")
    
    # Create analyzer
    analyzer = WorkloadPatternAnalyzer()
    
    # Generate cyclic workload
    cpu_data = generate_cyclic_workload(period=50)
    memory_data = generate_cyclic_workload(period=100, base=50, amplitude=20)
    disk_data = generate_cyclic_workload(period=200, base=60, amplitude=10)
    
    # Add data points
    start_time = datetime.now() - timedelta(minutes=len(cpu_data))
    for i in range(len(cpu_data)):
        timestamp = start_time + timedelta(seconds=i)
        analyzer.add_data_point(cpu_data[i], memory_data[i], disk_data[i], timestamp)
    
    # Get pattern
    pattern = analyzer.get_current_pattern()
    if pattern:
        print(f"Detected pattern: {pattern.pattern_type}")
        print(f"Confidence: {pattern.confidence:.2f}")
        if pattern.periodicity:
            print(f"Periodicity: {pattern.periodicity:.1f} seconds")
        if pattern.peak_times:
            print(f"Peak times: {', '.join(pattern.peak_times[:5])}")
    else:
        print("No pattern detected")
    
    # Get prediction
    prediction = analyzer.get_resource_prediction()
    if prediction:
        print(f"Prediction confidence: {prediction.confidence:.2f}")
        print(f"Recommended actions: {', '.join(prediction.recommended_actions)}")
        
        # Print some predicted values
        time_points = list(prediction.predicted_cpu.keys())[:5]
        for time_point in time_points:
            print(f"  {time_point}: CPU={prediction.predicted_cpu[time_point]:.1f}%, "
                 f"Memory={prediction.predicted_memory[time_point]:.1f}%")
    else:
        print("No prediction available")
    
    # Plot data and predictions if plotting is available
    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        x_hist = range(len(cpu_data))
        plt.plot(x_hist, cpu_data, 'b-', alpha=0.5, label='CPU History')
        plt.plot(x_hist, memory_data, 'g-', alpha=0.5, label='Memory History')
        
        # Plot predictions if available
        if prediction and prediction.predicted_cpu:
            x_pred = range(len(cpu_data), len(cpu_data) + len(prediction.predicted_cpu))
            y_pred_cpu = list(prediction.predicted_cpu.values())
            y_pred_mem = list(prediction.predicted_memory.values())
            
            plt.plot(x_pred, y_pred_cpu, 'b--', label='CPU Prediction')
            plt.plot(x_pred, y_pred_mem, 'g--', label='Memory Prediction')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Usage (%)')
        plt.title('Resource Usage History and Prediction')
        plt.legend()
        plt.grid(True)
        plt.savefig('workload_prediction.png')
        print("Plot saved as 'workload_prediction.png'")
    else:
        print("Plotting skipped (matplotlib not available)")


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
    # Test workload pattern analyzer
    test_workload_pattern_analyzer()
    
    # Test predictive resource allocation
    test_predictive_resource_allocation()