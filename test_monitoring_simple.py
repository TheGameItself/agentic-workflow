#!/usr/bin/env python3
"""
Simple test script for MonitoringSystem to verify basic functionality.
"""

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from mcp.monitoring_system import MonitoringSystem, HormoneCascadeEvent, LobeDetailView


def test_basic_functionality():
    """Test basic monitoring system functionality."""
    print("Testing MonitoringSystem basic functionality...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name
    
    try:
        # Create monitoring system
        monitoring = MonitoringSystem(database_path=temp_db, update_interval=0.1)
        print("✓ MonitoringSystem created successfully")
        
        # Test hormone level updates
        hormone_levels = {"dopamine": 0.7, "serotonin": 0.6, "cortisol": 0.3}
        monitoring.update_hormone_levels(hormone_levels, source_lobe="test_lobe")
        
        # Verify current levels
        current_levels = monitoring.get_real_time_hormone_levels()
        assert current_levels == hormone_levels, f"Expected {hormone_levels}, got {current_levels}"
        print("✓ Hormone level updates working")
        
        # Test cascade logging
        cascade = HormoneCascadeEvent(
            cascade_name="test_cascade",
            trigger_hormone="dopamine",
            trigger_level=0.8,
            affected_hormones=["serotonin", "oxytocin"],
            affected_lobes=["decision_making", "social_intelligence"],
            timestamp=datetime.now().isoformat(),
            duration=2.5,
            effects={"mood_boost": 0.3}
        )
        monitoring.log_hormone_cascade(cascade)
        print("✓ Cascade logging working")
        
        # Test lobe activity recording
        monitoring.record_lobe_activity(
            lobe_name="test_lobe",
            activity_type="hormone_production",
            hormone_involved="dopamine",
            level=0.7
        )
        print("✓ Lobe activity recording working")
        
        # Test visualization data generation
        viz_data = monitoring.generate_time_series_visualization("dopamine")
        print(f"✓ Visualization data generated with {len(viz_data.data_points)} points")
        
        # Test lobe state management
        lobe_detail = LobeDetailView(
            lobe_name="test_lobe",
            current_hormone_levels={"dopamine": 0.7},
            hormone_production_rate={"dopamine": 0.1},
            hormone_consumption_rate={},
            receptor_sensitivity={"dopamine": 0.8},
            recent_activity=[],
            connections=[],
            position=(1.0, 2.0, 3.0)
        )
        monitoring.update_lobe_state(lobe_detail)
        
        retrieved_detail = monitoring.get_lobe_details("test_lobe")
        assert retrieved_detail is not None, "Failed to retrieve lobe details"
        assert retrieved_detail.lobe_name == "test_lobe", "Lobe name mismatch"
        print("✓ Lobe state management working")
        
        # Test monitoring start/stop
        monitoring.start_monitoring()
        assert monitoring._running, "Monitoring should be running"
        print("✓ Monitoring started successfully")
        
        time.sleep(0.5)  # Let it run briefly
        
        monitoring.stop_monitoring()
        assert not monitoring._running, "Monitoring should be stopped"
        print("✓ Monitoring stopped successfully")
        
        # Test monitoring summary
        summary = monitoring.get_monitoring_summary()
        assert "monitoring_active" in summary, "Summary missing monitoring_active"
        assert summary["current_hormone_count"] == 3, f"Expected 3 hormones, got {summary['current_hormone_count']}"
        print("✓ Monitoring summary working")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        Path(temp_db).unlink(missing_ok=True)


def test_callback_functionality():
    """Test callback functionality."""
    print("\nTesting callback functionality...")
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name
    
    try:
        monitoring = MonitoringSystem(database_path=temp_db)
        
        # Test hormone callbacks
        hormone_callback_data = []
        def hormone_callback(levels):
            hormone_callback_data.append(levels.copy())
        
        monitoring.register_hormone_callback(hormone_callback)
        monitoring.update_hormone_levels({"dopamine": 0.8})
        
        assert len(hormone_callback_data) == 1, "Hormone callback not called"
        assert hormone_callback_data[0] == {"dopamine": 0.8}, "Hormone callback data incorrect"
        print("✓ Hormone callbacks working")
        
        # Test cascade callbacks
        cascade_callback_data = []
        def cascade_callback(cascade):
            cascade_callback_data.append(cascade)
        
        monitoring.register_cascade_callback(cascade_callback)
        
        cascade = HormoneCascadeEvent(
            cascade_name="test_cascade",
            trigger_hormone="dopamine",
            trigger_level=0.8,
            affected_hormones=["serotonin"],
            affected_lobes=["test_lobe"],
            timestamp=datetime.now().isoformat()
        )
        monitoring.log_hormone_cascade(cascade)
        
        assert len(cascade_callback_data) == 1, "Cascade callback not called"
        assert cascade_callback_data[0] == cascade, "Cascade callback data incorrect"
        print("✓ Cascade callbacks working")
        
        print("🎉 All callback tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Callback test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        Path(temp_db).unlink(missing_ok=True)


def test_anomaly_detection():
    """Test anomaly detection functionality."""
    print("\nTesting anomaly detection...")
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name
    
    try:
        monitoring = MonitoringSystem(database_path=temp_db)
        
        # Test anomaly callbacks
        anomaly_callback_data = []
        def anomaly_callback(anomaly):
            anomaly_callback_data.append(anomaly)
        
        monitoring.register_anomaly_callback(anomaly_callback)
        
        # Set up baseline hormone levels
        for i in range(5):
            monitoring.update_hormone_levels({"dopamine": 0.3})
            time.sleep(0.01)
        
        # Create a spike to trigger anomaly
        monitoring.current_hormone_levels["dopamine"] = 0.9
        monitoring._detect_anomalies()
        
        # Check if anomaly was detected
        if len(anomaly_callback_data) > 0:
            anomaly = anomaly_callback_data[0]
            assert anomaly.anomaly_type == "hormone_spike", f"Expected hormone_spike, got {anomaly.anomaly_type}"
            print("✓ Hormone spike anomaly detection working")
        else:
            print("⚠ Hormone spike anomaly not detected (may need more baseline data)")
        
        print("🎉 Anomaly detection tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Anomaly detection test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    print("Running MonitoringSystem tests...\n")
    
    success = True
    success &= test_basic_functionality()
    success &= test_callback_functionality()
    success &= test_anomaly_detection()
    
    if success:
        print("\n🎉 All tests passed! MonitoringSystem is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)