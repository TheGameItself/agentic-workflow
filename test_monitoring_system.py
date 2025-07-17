"""
Comprehensive tests for the MonitoringSystem class.

Tests monitoring accuracy, data integrity, real-time hormone tracking,
cascade logging, time-series visualization, and anomaly detection.
"""

import pytest
import sqlite3
import tempfile
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.mcp.monitoring_system import (
    MonitoringSystem, 
    HormoneCascadeEvent, 
    VisualizationData, 
    Anomaly, 
    LobeDetailView
)


class TestMonitoringSystem:
    """Test suite for MonitoringSystem class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def monitoring_system(self, temp_db_path):
        """Create a MonitoringSystem instance for testing."""
        return MonitoringSystem(
            database_path=temp_db_path,
            update_interval=0.1,  # Fast updates for testing
            max_history_entries=100
        )
    
    def test_initialization(self, monitoring_system, temp_db_path):
        """Test MonitoringSystem initialization."""
        assert monitoring_system.database_path == temp_db_path
        assert monitoring_system.update_interval == 0.1
        assert monitoring_system.max_history_entries == 100
        assert not monitoring_system._running
        assert len(monitoring_system.current_hormone_levels) == 0
        
        # Check database was created
        assert Path(temp_db_path).exists()
        
        # Verify database schema
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ["hormone_levels", "hormone_cascades", "lobe_activity", "anomalies"]
            for table in expected_tables:
                assert table in tables
    
    def test_hormone_level_updates(self, monitoring_system):
        """Test hormone level updates and storage."""
        # Test basic update
        hormone_levels = {"dopamine": 0.7, "serotonin": 0.6, "cortisol": 0.3}
        monitoring_system.update_hormone_levels(hormone_levels, source_lobe="test_lobe")
        
        # Verify current levels
        current_levels = monitoring_system.get_real_time_hormone_levels()
        assert current_levels == hormone_levels
        
        # Verify database storage
        with sqlite3.connect(monitoring_system.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT hormone_name, level, source_lobe FROM hormone_levels")
            rows = cursor.fetchall()
            
            assert len(rows) == 3
            stored_data = {row[0]: (row[1], row[2]) for row in rows}
            
            for hormone, level in hormone_levels.items():
                assert hormone in stored_data
                assert stored_data[hormone][0] == level
                assert stored_data[hormone][1] == "test_lobe"
    
    def test_hormone_level_callbacks(self, monitoring_system):
        """Test hormone level update callbacks."""
        callback_data = []
        
        def test_callback(levels):
            callback_data.append(levels.copy())
        
        monitoring_system.register_hormone_callback(test_callback)
        
        # Update hormone levels
        hormone_levels = {"dopamine": 0.8}
        monitoring_system.update_hormone_levels(hormone_levels)
        
        # Verify callback was called
        assert len(callback_data) == 1
        assert callback_data[0] == hormone_levels
    
    def test_cascade_logging(self, monitoring_system):
        """Test hormone cascade event logging."""
        cascade_event = HormoneCascadeEvent(
            cascade_name="test_cascade",
            trigger_hormone="dopamine",
            trigger_level=0.8,
            affected_hormones=["serotonin", "oxytocin"],
            affected_lobes=["decision_making", "social_intelligence"],
            timestamp=datetime.now().isoformat(),
            duration=2.5,
            effects={"mood_boost": 0.3},
            feedback_loops=["inhibition_loop"]
        )
        
        monitoring_system.log_hormone_cascade(cascade_event)
        
        # Verify cascade was stored
        with sqlite3.connect(monitoring_system.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cascade_name, trigger_hormone, trigger_level, 
                       affected_hormones, affected_lobes, duration, effects, feedback_loops
                FROM hormone_cascades
            """)
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == "test_cascade"
            assert row[1] == "dopamine"
            assert row[2] == 0.8
            assert json.loads(row[3]) == ["serotonin", "oxytocin"]
            assert json.loads(row[4]) == ["decision_making", "social_intelligence"]
            assert row[5] == 2.5
            assert json.loads(row[6]) == {"mood_boost": 0.3}
            assert json.loads(row[7]) == ["inhibition_loop"]
    
    def test_cascade_callbacks(self, monitoring_system):
        """Test cascade event callbacks."""
        callback_data = []
        
        def test_callback(cascade):
            callback_data.append(cascade)
        
        monitoring_system.register_cascade_callback(test_callback)
        
        # Log cascade
        cascade_event = HormoneCascadeEvent(
            cascade_name="test_cascade",
            trigger_hormone="dopamine",
            trigger_level=0.8,
            affected_hormones=["serotonin"],
            affected_lobes=["decision_making"],
            timestamp=datetime.now().isoformat()
        )
        monitoring_system.log_hormone_cascade(cascade_event)
        
        # Verify callback was called
        assert len(callback_data) == 1
        assert callback_data[0] == cascade_event
    
    def test_lobe_activity_recording(self, monitoring_system):
        """Test lobe activity recording."""
        monitoring_system.record_lobe_activity(
            lobe_name="test_lobe",
            activity_type="hormone_production",
            hormone_involved="dopamine",
            level=0.7,
            context={"task": "test_task"}
        )
        
        # Verify activity was recorded
        with sqlite3.connect(monitoring_system.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT lobe_name, activity_type, hormone_involved, level, context
                FROM lobe_activity
            """)
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == "test_lobe"
            assert row[1] == "hormone_production"
            assert row[2] == "dopamine"
            assert row[3] == 0.7
            assert json.loads(row[4]) == {"task": "test_task"}
    
    def test_time_series_visualization(self, monitoring_system):
        """Test time-series visualization data generation."""
        # Add some test data
        base_time = datetime.now()
        for i in range(5):
            timestamp = (base_time + timedelta(minutes=i)).isoformat()
            with sqlite3.connect(monitoring_system.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO hormone_levels (timestamp, hormone_name, level, source_lobe)
                    VALUES (?, ?, ?, ?)
                """, (timestamp, "dopamine", 0.5 + i * 0.1, "test_lobe"))
                conn.commit()
        
        # Generate visualization data
        start_time = base_time.isoformat()
        end_time = (base_time + timedelta(hours=1)).isoformat()
        
        viz_data = monitoring_system.generate_time_series_visualization(
            hormone_name="dopamine",
            start_time=start_time,
            end_time=end_time
        )
        
        # Verify visualization data
        assert isinstance(viz_data, VisualizationData)
        assert len(viz_data.data_points) == 5
        assert viz_data.metadata["hormone_name"] == "dopamine"
        assert viz_data.chart_type == "line"
        assert viz_data.time_range == (start_time, end_time)
        
        # Check data points
        for i, point in enumerate(viz_data.data_points):
            assert point["value"] == 0.5 + i * 0.1
            assert point["source_lobe"] == "test_lobe"
    
    def test_anomaly_detection_hormone_spike(self, monitoring_system):
        """Test anomaly detection for hormone spikes."""
        callback_data = []
        
        def anomaly_callback(anomaly):
            callback_data.append(anomaly)
        
        monitoring_system.register_anomaly_callback(anomaly_callback)
        
        # Set up baseline hormone levels
        for i in range(5):
            monitoring_system.update_hormone_levels({"dopamine": 0.3})
            time.sleep(0.01)  # Small delay to create history
        
        # Create a spike
        monitoring_system.current_hormone_levels["dopamine"] = 0.9
        
        # Trigger anomaly detection
        monitoring_system._detect_anomalies()
        
        # Verify anomaly was detected
        assert len(callback_data) > 0
        anomaly = callback_data[0]
        assert anomaly.anomaly_type == "hormone_spike"
        assert anomaly.component == "hormone_dopamine"
        assert "spike" in anomaly.description.lower()
    
    def test_anomaly_detection_hormone_drop(self, monitoring_system):
        """Test anomaly detection for hormone drops."""
        callback_data = []
        
        def anomaly_callback(anomaly):
            callback_data.append(anomaly)
        
        monitoring_system.register_anomaly_callback(anomaly_callback)
        
        # Set up baseline hormone levels
        for i in range(5):
            monitoring_system.update_hormone_levels({"serotonin": 0.8})
            time.sleep(0.01)
        
        # Create a drop
        monitoring_system.current_hormone_levels["serotonin"] = 0.1
        
        # Trigger anomaly detection
        monitoring_system._detect_anomalies()
        
        # Verify anomaly was detected
        assert len(callback_data) > 0
        anomaly = callback_data[0]
        assert anomaly.anomaly_type == "hormone_drop"
        assert anomaly.component == "hormone_serotonin"
        assert "drop" in anomaly.description.lower()
    
    def test_anomaly_detection_excessive_cascades(self, monitoring_system):
        """Test anomaly detection for excessive cascades."""
        callback_data = []
        
        def anomaly_callback(anomaly):
            callback_data.append(anomaly)
        
        monitoring_system.register_anomaly_callback(anomaly_callback)
        
        # Create many cascades in a short time
        current_time = datetime.now()
        for i in range(10):  # More than threshold of 5
            cascade = HormoneCascadeEvent(
                cascade_name=f"cascade_{i}",
                trigger_hormone="dopamine",
                trigger_level=0.8,
                affected_hormones=["serotonin"],
                affected_lobes=["test_lobe"],
                timestamp=current_time.isoformat()
            )
            monitoring_system.active_cascades.append(cascade)
        
        # Trigger anomaly detection
        monitoring_system._detect_anomalies()
        
        # Verify anomaly was detected
        assert len(callback_data) > 0
        anomaly = callback_data[0]
        assert anomaly.anomaly_type == "excessive_cascades"
        assert anomaly.component == "hormone_system"
        assert "excessive" in anomaly.description.lower()
    
    def test_lobe_state_management(self, monitoring_system):
        """Test lobe state updates and retrieval."""
        lobe_detail = LobeDetailView(
            lobe_name="test_lobe",
            current_hormone_levels={"dopamine": 0.7, "serotonin": 0.6},
            hormone_production_rate={"dopamine": 0.1},
            hormone_consumption_rate={"serotonin": 0.05},
            receptor_sensitivity={"dopamine": 0.8, "serotonin": 0.9},
            recent_activity=[{"type": "production", "hormone": "dopamine"}],
            connections=["other_lobe"],
            position=(1.0, 2.0, 3.0)
        )
        
        # Update lobe state
        monitoring_system.update_lobe_state(lobe_detail)
        
        # Retrieve lobe state
        retrieved_detail = monitoring_system.get_lobe_details("test_lobe")
        
        assert retrieved_detail is not None
        assert retrieved_detail.lobe_name == "test_lobe"
        assert retrieved_detail.current_hormone_levels == {"dopamine": 0.7, "serotonin": 0.6}
        assert retrieved_detail.position == (1.0, 2.0, 3.0)
    
    def test_monitoring_start_stop(self, monitoring_system):
        """Test monitoring system start and stop functionality."""
        # Initially not running
        assert not monitoring_system._running
        
        # Start monitoring
        monitoring_system.start_monitoring()
        assert monitoring_system._running
        assert monitoring_system._monitor_thread is not None
        
        # Stop monitoring
        monitoring_system.stop_monitoring()
        assert not monitoring_system._running
    
    def test_data_cleanup(self, monitoring_system):
        """Test old data cleanup functionality."""
        # Add old data
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        
        with sqlite3.connect(monitoring_system.database_path) as conn:
            cursor = conn.cursor()
            
            # Add old hormone level
            cursor.execute("""
                INSERT INTO hormone_levels (timestamp, hormone_name, level)
                VALUES (?, ?, ?)
            """, (old_time, "dopamine", 0.5))
            
            # Add old lobe activity
            cursor.execute("""
                INSERT INTO lobe_activity (timestamp, lobe_name, activity_type)
                VALUES (?, ?, ?)
            """, (old_time, "test_lobe", "test_activity"))
            
            conn.commit()
        
        # Run cleanup
        monitoring_system._cleanup_old_data()
        
        # Verify old data was removed
        with sqlite3.connect(monitoring_system.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM hormone_levels WHERE timestamp = ?", (old_time,))
            assert cursor.fetchone()[0] == 0
            
            cursor.execute("SELECT COUNT(*) FROM lobe_activity WHERE timestamp = ?", (old_time,))
            assert cursor.fetchone()[0] == 0
    
    def test_monitoring_summary(self, monitoring_system):
        """Test monitoring summary generation."""
        # Add some test data
        monitoring_system.update_hormone_levels({"dopamine": 0.7})
        
        lobe_detail = LobeDetailView(
            lobe_name="test_lobe",
            current_hormone_levels={},
            hormone_production_rate={},
            hormone_consumption_rate={},
            receptor_sensitivity={},
            recent_activity=[],
            connections=[],
            position=(0, 0, 0)
        )
        monitoring_system.update_lobe_state(lobe_detail)
        
        # Get summary
        summary = monitoring_system.get_monitoring_summary()
        
        assert "monitoring_active" in summary
        assert summary["current_hormone_count"] == 1
        assert summary["monitored_lobe_count"] == 1
        assert "database_path" in summary
        assert "update_interval" in summary
        assert "last_update" in summary
    
    def test_thread_safety(self, monitoring_system):
        """Test thread safety of monitoring operations."""
        import threading
        
        results = []
        errors = []
        
        def update_hormones(thread_id):
            try:
                for i in range(10):
                    monitoring_system.update_hormone_levels({
                        f"hormone_{thread_id}": 0.5 + i * 0.1
                    })
                    time.sleep(0.001)
                results.append(thread_id)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_hormones, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0
        assert len(results) == 5
        
        # Verify data integrity
        levels = monitoring_system.get_real_time_hormone_levels()
        assert len(levels) == 5  # One hormone per thread


def test_integration_with_hormone_system():
    """Test integration with hormone system components."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name
    
    try:
        monitoring = MonitoringSystem(database_path=temp_db)
        
        # Simulate hormone system integration
        monitoring.start_monitoring()
        
        # Update hormone levels as if from hormone system
        monitoring.update_hormone_levels({
            "dopamine": 0.8,
            "serotonin": 0.7,
            "cortisol": 0.4
        }, source_lobe="task_management")
        
        # Log cascade as if from hormone system
        cascade = HormoneCascadeEvent(
            cascade_name="reward_cascade",
            trigger_hormone="dopamine",
            trigger_level=0.8,
            affected_hormones=["serotonin", "oxytocin"],
            affected_lobes=["decision_making", "social_intelligence"],
            timestamp=datetime.now().isoformat(),
            duration=1.5
        )
        monitoring.log_hormone_cascade(cascade)
        
        # Verify data was recorded
        levels = monitoring.get_real_time_hormone_levels()
        assert levels["dopamine"] == 0.8
        assert levels["serotonin"] == 0.7
        assert levels["cortisol"] == 0.4
        
        # Generate visualization
        viz_data = monitoring.generate_time_series_visualization("dopamine")
        assert len(viz_data.data_points) >= 1
        
        monitoring.stop_monitoring()
        
    finally:
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])