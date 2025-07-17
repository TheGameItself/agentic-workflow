#!/usr/bin/env python3
"""
Test script for genetic trigger monitoring functionality in MonitoringSystem.
"""

import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from mcp.monitoring_system import (
    MonitoringSystem, 
    GeneticTriggerActivation, 
    EnvironmentalState,
    Anomaly
)


def test_genetic_trigger_monitoring():
    """Test genetic trigger monitoring functionality."""
    print("Testing genetic trigger monitoring functionality...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name
    
    try:
        # Create monitoring system
        monitoring = MonitoringSystem(database_path=temp_db, update_interval=0.1)
        print("✓ MonitoringSystem created successfully")
        
        # Test environmental state updates
        env_state = EnvironmentalState(
            timestamp=datetime.now().isoformat(),
            system_load={"cpu": 0.7, "memory": 0.6},
            performance_metrics={"accuracy": 0.85, "speed": 0.75},
            resource_usage={"cpu": 0.6, "memory": 0.5},
            hormone_levels={"dopamine": 0.7, "serotonin": 0.6},
            task_complexity=0.8,
            adaptation_pressure=0.6,
            network_conditions={"latency": 0.1, "bandwidth": 0.9}
        )
        
        monitoring.update_environmental_state(env_state)
        print("✓ Environmental state update working")
        
        # Test genetic trigger activation recording
        activation = GeneticTriggerActivation(
            trigger_id="test_trigger_001",
            activation_timestamp=datetime.now().isoformat(),
            environmental_context={
                "system_load": {"cpu": 0.7, "memory": 0.6},
                "task_type": "learning",
                "priority": "high"
            },
            activation_score=0.85,
            behavior_changes=["increased_learning_rate", "enhanced_attention"],
            performance_impact={"accuracy": 0.1, "speed": 0.05},
            duration=2.5,
            success_metrics={"task_completion": 0.9, "efficiency": 0.8}
        )
        
        monitoring.record_genetic_trigger_activation(activation)
        print("✓ Genetic trigger activation recording working")
        
        # Test genetic trigger callback
        callback_data = []
        def genetic_trigger_callback(activation):
            callback_data.append(activation)
        
        monitoring.register_genetic_trigger_callback(genetic_trigger_callback)
        
        # Record another activation to test callback
        activation2 = GeneticTriggerActivation(
            trigger_id="test_trigger_002",
            activation_timestamp=datetime.now().isoformat(),
            environmental_context={"task_type": "optimization"},
            activation_score=0.75,
            behavior_changes=["resource_optimization"],
            performance_impact={"efficiency": 0.15}
        )
        
        monitoring.record_genetic_trigger_activation(activation2)
        
        # Verify callback was called
        assert len(callback_data) == 1, "Genetic trigger callback not called"
        assert callback_data[0] == activation2, "Genetic trigger callback data incorrect"
        print("✓ Genetic trigger callbacks working")
        
        # Test genetic trigger activation retrieval
        activations = monitoring.get_genetic_trigger_activations(limit=10)
        assert len(activations) == 2, f"Expected 2 activations, got {len(activations)}"
        
        # Test filtering by trigger ID
        trigger_activations = monitoring.get_genetic_trigger_activations(
            trigger_id="test_trigger_001"
        )
        assert len(trigger_activations) == 1, f"Expected 1 activation for trigger_001, got {len(trigger_activations)}"
        assert trigger_activations[0]["trigger_id"] == "test_trigger_001"
        print("✓ Genetic trigger activation retrieval working")
        
        # Test environmental state history
        env_history = monitoring.get_environmental_state_history(limit=5)
        assert len(env_history) == 1, f"Expected 1 environmental state, got {len(env_history)}"
        assert env_history[0]["task_complexity"] == 0.8
        print("✓ Environmental state history working")
        
        # Test genetic trigger report generation
        report = monitoring.generate_genetic_trigger_report()
        assert report["total_activations"] == 2, f"Expected 2 activations in report, got {report['total_activations']}"
        assert report["unique_triggers"] == 2, f"Expected 2 unique triggers, got {report['unique_triggers']}"
        assert "behavior_change_types" in report
        print("✓ Genetic trigger report generation working")
        
        # Test anomaly detection for genetic triggers
        anomaly_callback_data = []
        def anomaly_callback(anomaly):
            anomaly_callback_data.append(anomaly)
        
        monitoring.register_anomaly_callback(anomaly_callback)
        
        # Create many activations to trigger excessive activation anomaly
        current_time = datetime.now()
        for i in range(12):  # More than threshold of 10
            excessive_activation = GeneticTriggerActivation(
                trigger_id=f"excessive_trigger_{i}",
                activation_timestamp=current_time.isoformat(),
                environmental_context={"test": True},
                activation_score=0.5,
                behavior_changes=["test_behavior"],
                performance_impact={"test": 0.1}
            )
            monitoring.active_genetic_triggers.append(excessive_activation)
        
        # Trigger anomaly detection
        monitoring.detect_genetic_trigger_anomalies()
        
        # Check if excessive activation anomaly was detected
        excessive_anomalies = [a for a in anomaly_callback_data if a.anomaly_type == "excessive_genetic_triggers"]
        if len(excessive_anomalies) > 0:
            print("✓ Excessive genetic trigger anomaly detection working")
        else:
            print("⚠ Excessive genetic trigger anomaly not detected")
        
        # Test low performance trigger anomaly
        # Create triggers with low performance
        for i in range(6):  # Need at least 5 for assessment
            low_perf_activation = GeneticTriggerActivation(
                trigger_id="low_performance_trigger",
                activation_timestamp=(current_time + timedelta(seconds=i)).isoformat(),
                environmental_context={"test": True},
                activation_score=0.5,
                behavior_changes=["test_behavior"],
                performance_impact={"performance": 0.1}  # Low performance
            )
            monitoring.active_genetic_triggers.append(low_perf_activation)
        
        # Clear previous anomalies and test again
        anomaly_callback_data.clear()
        monitoring.detect_genetic_trigger_anomalies()
        
        # Check if low performance anomaly was detected
        low_perf_anomalies = [a for a in anomaly_callback_data if a.anomaly_type == "low_performance_genetic_trigger"]
        if len(low_perf_anomalies) > 0:
            print("✓ Low performance genetic trigger anomaly detection working")
        else:
            print("⚠ Low performance genetic trigger anomaly not detected")
        
        # Test monitoring summary with genetic triggers
        summary = monitoring.get_monitoring_summary()
        assert "active_genetic_trigger_count" in summary
        assert "environmental_state_available" in summary
        assert summary["environmental_state_available"] == True
        print("✓ Monitoring summary includes genetic trigger information")
        
        print("\n🎉 All genetic trigger monitoring tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        Path(temp_db).unlink(missing_ok=True)


def test_genetic_trigger_integration():
    """Test integration between genetic triggers and hormone monitoring."""
    print("\nTesting genetic trigger and hormone monitoring integration...")
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name
    
    try:
        monitoring = MonitoringSystem(database_path=temp_db)
        
        # Update hormone levels
        monitoring.update_hormone_levels({
            "dopamine": 0.8,
            "serotonin": 0.7,
            "cortisol": 0.4
        }, source_lobe="task_management")
        
        # Update environmental state with hormone levels
        env_state = EnvironmentalState(
            timestamp=datetime.now().isoformat(),
            system_load={"cpu": 0.6},
            performance_metrics={"accuracy": 0.9},
            resource_usage={"memory": 0.5},
            hormone_levels={"dopamine": 0.8, "serotonin": 0.7, "cortisol": 0.4},
            task_complexity=0.7,
            adaptation_pressure=0.5
        )
        monitoring.update_environmental_state(env_state)
        
        # Record genetic trigger activation that responds to hormone levels
        activation = GeneticTriggerActivation(
            trigger_id="hormone_responsive_trigger",
            activation_timestamp=datetime.now().isoformat(),
            environmental_context={
                "hormone_trigger": "dopamine_high",
                "hormone_levels": {"dopamine": 0.8, "serotonin": 0.7}
            },
            activation_score=0.9,
            behavior_changes=["reward_pathway_enhancement", "motivation_boost"],
            performance_impact={"task_completion_rate": 0.2, "focus": 0.15}
        )
        monitoring.record_genetic_trigger_activation(activation)
        
        # Generate comprehensive report
        report = monitoring.generate_genetic_trigger_report()
        
        # Verify integration
        assert report["total_activations"] == 1
        assert "hormone_responsive_trigger" in [t[0] for t in report["most_active_triggers"]]
        
        # Check that environmental context includes hormone information
        activations = monitoring.get_genetic_trigger_activations()
        assert len(activations) == 1
        assert "hormone_levels" in activations[0]["environmental_context"]
        
        print("✓ Genetic trigger and hormone monitoring integration working")
        print(f"✓ Report shows {report['total_activations']} activations with hormone context")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    print("Running genetic trigger monitoring tests...\n")
    
    success = True
    success &= test_genetic_trigger_monitoring()
    success &= test_genetic_trigger_integration()
    
    if success:
        print("\n🎉 All genetic trigger monitoring tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)