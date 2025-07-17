#!/usr/bin/env python3
"""
Test script for the MonitoringSystem visualization and reporting capabilities.

This script demonstrates the visualization data generation, performance comparison
reporting, and trend analysis features of the MonitoringSystem.
"""

import sys
import os
import json
from datetime import datetime, timedelta
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.mcp.monitoring_system_visualization_integration import MonitoringSystemWithVisualization
from src.mcp.monitoring_system_visualization import VisualizationOptions


def generate_test_data(monitoring):
    """Generate test data for visualization and reporting."""
    print("Generating test data...")
    
    # Generate hormone levels
    hormones = ["dopamine", "serotonin", "cortisol", "adrenaline", "oxytocin"]
    lobes = ["task_management", "decision_making", "memory", "pattern_recognition"]
    
    # Generate data for the past 24 hours
    now = datetime.now()
    for i in range(24):
        for j in range(10):  # 10 data points per hour
            for hormone in hormones:
                # Create some patterns in the data
                if hormone == "dopamine":
                    level = 0.7 + 0.2 * (i % 10) / 10.0  # Oscillating pattern
                elif hormone == "serotonin":
                    level = 0.5 + 0.01 * i  # Increasing trend
                elif hormone == "cortisol":
                    level = 0.8 - 0.005 * i  # Decreasing trend
                elif hormone == "adrenaline":
                    level = 0.3 + 0.4 * (i > 12)  # Step function (higher in second half)
                else:  # oxytocin
                    level = 0.6 + 0.1 * random.random()  # Random fluctuation
                
                # Add some noise
                level += 0.05 * (random.random() - 0.5)
                level = max(0.0, min(1.0, level))  # Clamp to [0, 1]
                
                source_lobe = lobes[random.randint(0, len(lobes) - 1)]
                
                monitoring.update_hormone_levels(
                    {hormone: level},
                    source_lobe=source_lobe
                )
    
    # Generate hormone cascades
    cascade_names = ["reward_cascade", "stress_response", "learning_enhancement"]
    trigger_hormones = ["dopamine", "cortisol", "serotonin"]
    
    for i in range(10):
        timestamp = (now - timedelta(hours=i)).isoformat()
        cascade_idx = i % len(cascade_names)
        
        monitoring.log_hormone_cascade({
            "cascade_name": cascade_names[cascade_idx],
            "trigger_hormone": trigger_hormones[cascade_idx],
            "trigger_level": 0.7 + 0.1 * (i % 3),
            "affected_hormones": ["dopamine", "serotonin"] if cascade_idx != 0 else ["serotonin", "cortisol"],
            "affected_lobes": ["task_management", "decision_making"] if cascade_idx != 2 else ["memory", "decision_making"],
            "timestamp": timestamp,
            "duration": 1.5 + 0.5 * (i % 3),
            "effects": {"mood_boost": 0.3, "confidence_increase": 0.2},
            "feedback_loops": ["positive_reinforcement", "stress_reduction"]
        })
    
    # Generate performance metrics
    components = ["hormone_calculator", "diffusion_engine", "receptor_sensitivity"]
    implementations = ["algorithmic", "neural"]
    
    for i in range(48):  # 48 hours of data
        for component in components:
            # Algorithmic implementation metrics
            monitoring.record_performance_metrics(
                component=component,
                implementation="algorithmic",
                metrics={
                    "accuracy": 0.85 + 0.01 * (i % 5),  # Slight oscillation
                    "latency": 0.15 - 0.001 * i,  # Slight improvement
                    "resource_usage": 0.3 + 0.002 * i  # Slight increase
                }
            )
            
            # Neural implementation metrics
            monitoring.record_performance_metrics(
                component=component,
                implementation="neural",
                metrics={
                    "accuracy": 0.9 + 0.005 * min(i, 20),  # Increasing then plateau
                    "latency": 0.2 - 0.005 * min(i, 30),  # Faster improvement then plateau
                    "resource_usage": 0.5 + 0.005 * min(i, 15)  # Faster increase then plateau
                }
            )
    
    # Generate implementation switches
    for i in range(3):
        component = components[i % len(components)]
        
        monitoring.record_implementation_switch(
            component=component,
            old_implementation="algorithmic",
            new_implementation="neural",
            reason=f"Neural implementation showed better accuracy for {component}",
            performance_comparison={
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
            },
            switch_trigger="performance",
            confidence_score=0.8,
            expected_improvement={
                "accuracy": 0.1,
                "latency": 0.05
            }
        )
    
    print("Test data generation complete.")


def test_multi_hormone_visualization(monitoring):
    """Test multi-hormone visualization."""
    print("\nTesting multi-hormone visualization...")
    
    # Generate visualization with default options
    viz_data = monitoring.generate_multi_hormone_visualization(
        hormone_names=["dopamine", "serotonin", "cortisol"]
    )
    
    print(f"Generated visualization with {len(viz_data['hormone_data'])} hormones")
    print(f"Time points: {len(viz_data['time_points'])}")
    
    # Generate visualization with custom options
    options = VisualizationOptions(
        chart_type="area",
        smoothing_factor=0.3,
        include_anomalies=True,
        include_forecast=True
    )
    
    viz_data = monitoring.generate_multi_hormone_visualization(
        hormone_names=["dopamine", "serotonin"],
        options=options
    )
    
    print(f"Generated visualization with custom options:")
    print(f"Chart type: {viz_data['chart_type']}")
    print(f"Smoothing factor: {viz_data['options']['smoothing_factor']}")
    print(f"Forecasts: {len(viz_data['forecasts'])}")
    print(f"Anomalies: {len(viz_data['anomalies'])}")


def test_cascade_visualization(monitoring):
    """Test cascade visualization."""
    print("\nTesting cascade visualization...")
    
    viz_data = monitoring.generate_cascade_visualization()
    
    print(f"Generated cascade visualization with {len(viz_data['cascades'])} cascades")
    print(f"Network nodes: {len(viz_data['network']['nodes'])}")
    print(f"Network links: {len(viz_data['network']['links'])}")
    print(f"Unique hormones: {viz_data['metadata']['unique_hormones']}")
    print(f"Unique lobes: {viz_data['metadata']['unique_lobes']}")
    
    # Test with specific cascade name
    viz_data = monitoring.generate_cascade_visualization(
        cascade_name="reward_cascade"
    )
    
    print(f"Generated filtered cascade visualization with {len(viz_data['cascades'])} cascades")


def test_performance_comparison_report(monitoring):
    """Test performance comparison report."""
    print("\nTesting performance comparison report...")
    
    report = monitoring.generate_performance_comparison_report("hormone_calculator")
    
    if report:
        print(f"Generated performance report for {report.component}")
        print(f"Winner implementation: {report.winner}")
        print(f"Confidence score: {report.confidence_score:.2f}")
        print(f"Improvement percentages: {report.improvement_percentage}")
        print(f"Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    else:
        print("No performance data available for comparison")


def test_system_performance_report(monitoring):
    """Test system performance report."""
    print("\nTesting system performance report...")
    
    report = monitoring.generate_system_performance_report()
    
    print(f"Generated system performance report with {report['component_count']} components")
    print(f"System averages: {report['system_averages']}")
    print(f"Significant improvements: {len(report['significant_improvements'])}")
    print(f"Concerning trends: {len(report['concerning_trends'])}")
    print(f"System recommendations:")
    for rec in report['system_recommendations']:
        print(f"  - {rec}")


def main():
    """Main test function."""
    print("Testing MonitoringSystem visualization and reporting capabilities...")
    
    # Create monitoring system
    monitoring = MonitoringSystemWithVisualization()
    
    try:
        # Start monitoring
        monitoring.start_monitoring()
        
        # Generate test data
        generate_test_data(monitoring)
        
        # Test visualization and reporting
        test_multi_hormone_visualization(monitoring)
        test_cascade_visualization(monitoring)
        test_performance_comparison_report(monitoring)
        test_system_performance_report(monitoring)
        
        print("\nAll tests completed successfully!")
        
    finally:
        # Stop monitoring
        monitoring.stop_monitoring()


if __name__ == "__main__":
    main()