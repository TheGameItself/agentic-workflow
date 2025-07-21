"""
MonitoringSystem: Real-time hormone level monitoring and visualization system.

This module implements comprehensive monitoring capabilities for the hormone system,
including real-time tracking, cascade logging, time-series data collection, and
visualization support for the MCP brain-inspired architecture.

References:
- Requirements 4.1, 4.2, 4.6 from mcp-system-upgrade spec
- idea.txt (brain-inspired architecture)
- cross-implementation.md (hormone system integration)
"""

import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import statistics
from pathlib import Path
from enum import Enum


@dataclass
class HormoneCascadeEvent:
    """Represents a hormone cascade event for logging and visualization."""
    cascade_name: str
    trigger_hormone: str
    trigger_level: float
    affected_hormones: List[str]
    affected_lobes: List[str]
    timestamp: str
    duration: float = 0.0
    effects: Dict[str, Any] = field(default_factory=dict)
    feedback_loops: List[str] = field(default_factory=list)


@dataclass
class VisualizationData:
    """Data structure for visualization components."""
    data_points: List[Dict[str, Any]]
    labels: List[str]
    metadata: Dict[str, Any]
    chart_type: str = "line"
    time_range: Optional[Tuple[str, str]] = None


@dataclass
class Anomaly:
    """Represents an anomaly detected in system behavior."""
    anomaly_type: str
    component: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class LobeDetailView:
    """Detailed view of a specific lobe's hormone activity."""
    lobe_name: str
    current_hormone_levels: Dict[str, float]
    hormone_production_rate: Dict[str, float]
    hormone_consumption_rate: Dict[str, float]
    receptor_sensitivity: Dict[str, float]
    recent_activity: List[Dict[str, Any]]
    connections: List[str]
    position: Tuple[float, float, float]


@dataclass
class GeneticTriggerActivation:
    """Represents a genetic trigger activation event."""
    trigger_id: str
    activation_timestamp: str
    environmental_context: Dict[str, Any]
    activation_score: float
    behavior_changes: List[str]
    performance_impact: Dict[str, float]
    duration: float = 0.0
    success_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnvironmentalState:
    """Represents the environmental state for genetic trigger evaluation."""
    timestamp: str
    system_load: Dict[str, float]
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    hormone_levels: Dict[str, float]
    task_complexity: float
    adaptation_pressure: float
    network_conditions: Dict[str, float] = field(default_factory=dict)


@dataclass
class ImplementationSwitchEvent:
    """Represents an implementation switch event."""
    component: str
    old_implementation: str
    new_implementation: str
    switch_timestamp: str
    reason: str
    performance_comparison: Dict[str, Dict[str, float]]
    switch_trigger: str  # "performance", "failure", "manual", "scheduled"
    confidence_score: float = 0.0
    expected_improvement: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance report for implementations."""
    component: str
    report_timestamp: str
    time_range: Tuple[str, str]
    implementations: Dict[str, Dict[str, Any]]
    active_implementation: str
    switch_history: List[Dict[str, Any]]
    performance_trends: Dict[str, List[float]]
    recommendations: List[str]
    efficiency_metrics: Dict[str, float]


class MonitoringSystem:
    """
    Comprehensive monitoring system for hormone levels, cascades, and brain state.
    
    Provides real-time tracking, historical analysis, anomaly detection, and
    visualization capabilities for the MCP hormone system.
    """
    
    def __init__(self, 
                 database_path: str = "data/monitoring_system.db",
                 update_interval: float = 1.0,
                 max_history_entries: int = 10000):
        """
        Initialize the monitoring system.
        
        Args:
            database_path: Path to SQLite database for persistent storage
            update_interval: How often to update monitoring data (seconds)
            max_history_entries: Maximum number of historical entries to keep
        """
        self.database_path = database_path
        self.update_interval = update_interval
        self.max_history_entries = max_history_entries
        
        # Initialize logging
        self.logger = logging.getLogger("MonitoringSystem")
        
        # Current state tracking
        self.current_hormone_levels: Dict[str, float] = {}
        self.current_lobe_states: Dict[str, LobeDetailView] = {}
        self.active_cascades: List[HormoneCascadeEvent] = []
        self.active_genetic_triggers: List[GeneticTriggerActivation] = []
        self.current_environmental_state: Optional[EnvironmentalState] = None
        self.implementation_switches: List[ImplementationSwitchEvent] = []
        self.active_implementations: Dict[str, str] = {}  # component -> implementation
        
        # Monitoring callbacks
        self.hormone_callbacks: List[Callable[[Dict[str, float]], None]] = []
        self.cascade_callbacks: List[Callable[[HormoneCascadeEvent], None]] = []
        self.anomaly_callbacks: List[Callable[[Anomaly], None]] = []
        self.genetic_trigger_callbacks: List[Callable[[GeneticTriggerActivation], None]] = []
        self.implementation_switch_callbacks: List[Callable[[ImplementationSwitchEvent], None]] = []
        
        # Anomaly detection parameters
        self.anomaly_thresholds = {
            "hormone_spike": 0.8,  # Threshold for sudden hormone increases
            "hormone_drop": 0.2,   # Threshold for sudden hormone decreases
            "cascade_frequency": 5,  # Max cascades per minute
            "lobe_silence": 300,   # Seconds of no activity before flagging
        }
        
        # Threading control
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info("MonitoringSystem initialized")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for persistent monitoring data storage."""
        # Ensure data directory exists
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Hormone levels table
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
            
            # Hormone cascades table
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
            
            # Lobe activity table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lobe_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    lobe_name TEXT NOT NULL,
                    activity_type TEXT NOT NULL,
                    hormone_involved TEXT,
                    level REAL,
                    context TEXT
                )
            """)
            
            # Anomalies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    data TEXT,
                    suggested_actions TEXT
                )
            """)
            
            # Genetic trigger activations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS genetic_trigger_activations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_id TEXT NOT NULL,
                    activation_timestamp TEXT NOT NULL,
                    environmental_context TEXT NOT NULL,
                    activation_score REAL NOT NULL,
                    behavior_changes TEXT,
                    performance_impact TEXT,
                    duration REAL,
                    success_metrics TEXT
                )
            """)
            
            # Environmental states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS environmental_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    system_load TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    resource_usage TEXT NOT NULL,
                    hormone_levels TEXT NOT NULL,
                    task_complexity REAL NOT NULL,
                    adaptation_pressure REAL NOT NULL,
                    network_conditions TEXT
                )
            """)
            
            # Implementation switches table
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
            
            # Performance metrics table
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
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hormone_timestamp ON hormone_levels(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cascade_timestamp ON hormone_cascades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lobe_timestamp ON lobe_activity(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomalies(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_genetic_trigger_timestamp ON genetic_trigger_activations(activation_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_environmental_timestamp ON environmental_states(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_implementation_switch_timestamp ON implementation_switches(switch_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_component ON performance_metrics(component, implementation)")
            
            conn.commit()
            
        self.logger.info("Database initialized successfully")
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self._running:
            self.logger.warning("Monitoring system is already running")
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        if not self._running:
            self.logger.warning("Monitoring system is not running")
            return
            
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            
        self.logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while self._running:
            try:
                # Update monitoring data
                self._update_monitoring_data()
                
                # Check for anomalies
                self._detect_anomalies()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_monitoring_data(self) -> None:
        """Update monitoring data from connected systems."""
        # This method would be called by external systems to update data
        # For now, it's a placeholder that maintains the monitoring loop
        pass
    
    def get_real_time_hormone_levels(self) -> Dict[str, float]:
        """
        Get current real-time hormone levels.
        
        Returns:
            Dictionary mapping hormone names to their current levels
        """
        with self._lock:
            return self.current_hormone_levels.copy()
    
    def update_hormone_levels(self, 
                            hormone_levels: Dict[str, float], 
                            source_lobe: Optional[str] = None,
                            context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update hormone levels and store in database.
        
        Args:
            hormone_levels: Dictionary of hormone names to levels
            source_lobe: Optional source lobe name
            context: Optional context information
        """
        timestamp = datetime.now().isoformat()
        
        with self._lock:
            # Update current levels
            self.current_hormone_levels.update(hormone_levels)
            
            # Store in database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                for hormone, level in hormone_levels.items():
                    cursor.execute("""
                        INSERT INTO hormone_levels 
                        (timestamp, hormone_name, level, source_lobe, context)
                        VALUES (?, ?, ?, ?, ?)
                    """, (timestamp, hormone, level, source_lobe, 
                         json.dumps(context) if context else None))
                
                conn.commit()
        
        # Notify callbacks
        for callback in self.hormone_callbacks:
            try:
                callback(hormone_levels)
            except Exception as e:
                self.logger.error(f"Error in hormone callback: {e}")
        
        self.logger.debug(f"Updated hormone levels: {hormone_levels}")
    
    def log_hormone_cascade(self, cascade_event: HormoneCascadeEvent) -> None:
        """
        Log a hormone cascade event.
        
        Args:
            cascade_event: The cascade event to log
        """
        with self._lock:
            # Add to active cascades
            self.active_cascades.append(cascade_event)
            
            # Store in database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO hormone_cascades 
                    (timestamp, cascade_name, trigger_hormone, trigger_level,
                     affected_hormones, affected_lobes, duration, effects, feedback_loops)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cascade_event.timestamp,
                    cascade_event.cascade_name,
                    cascade_event.trigger_hormone,
                    cascade_event.trigger_level,
                    json.dumps(cascade_event.affected_hormones),
                    json.dumps(cascade_event.affected_lobes),
                    cascade_event.duration,
                    json.dumps(cascade_event.effects),
                    json.dumps(cascade_event.feedback_loops)
                ))
                
                conn.commit()
        
        # Notify callbacks
        for callback in self.cascade_callbacks:
            try:
                callback(cascade_event)
            except Exception as e:
                self.logger.error(f"Error in cascade callback: {e}")
        
        self.logger.info(f"Logged hormone cascade: {cascade_event.cascade_name}")
    
    def record_lobe_activity(self, 
                           lobe_name: str, 
                           activity_type: str,
                           hormone_involved: Optional[str] = None,
                           level: Optional[float] = None,
                           context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record lobe activity for monitoring.
        
        Args:
            lobe_name: Name of the lobe
            activity_type: Type of activity (e.g., "hormone_production", "hormone_consumption")
            hormone_involved: Optional hormone name
            level: Optional hormone level
            context: Optional context information
        """
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO lobe_activity 
                (timestamp, lobe_name, activity_type, hormone_involved, level, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, lobe_name, activity_type, hormone_involved, level,
                 json.dumps(context) if context else None))
            
            conn.commit()
        
        self.logger.debug(f"Recorded lobe activity: {lobe_name} - {activity_type}")
    
    def generate_time_series_visualization(self, 
                                         hormone_name: str,
                                         start_time: Optional[str] = None,
                                         end_time: Optional[str] = None,
                                         resolution: str = "minute") -> VisualizationData:
        """
        Generate time-series visualization data for a hormone.
        
        Args:
            hormone_name: Name of the hormone
            start_time: Start time (ISO format), defaults to 1 hour ago
            end_time: End time (ISO format), defaults to now
            resolution: Time resolution ("second", "minute", "hour")
            
        Returns:
            VisualizationData object with time-series data
        """
        if not end_time:
            end_time = datetime.now().isoformat()
        if not start_time:
            start_time = (datetime.now() - timedelta(hours=1)).isoformat()
        
        # Query database for hormone data
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, level, source_lobe
                FROM hormone_levels
                WHERE hormone_name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (hormone_name, start_time, end_time))
            
            rows = cursor.fetchall()
        
        # Process data points
        data_points = []
        for timestamp, level, source_lobe in rows:
            data_points.append({
                "timestamp": timestamp,
                "value": level,
                "source_lobe": source_lobe
            })
        
        # Generate labels based on resolution
        labels = [point["timestamp"] for point in data_points]
        
        return VisualizationData(
            data_points=data_points,
            labels=labels,
            metadata={
                "hormone_name": hormone_name,
                "start_time": start_time,
                "end_time": end_time,
                "resolution": resolution,
                "total_points": len(data_points)
            },
            chart_type="line",
            time_range=(start_time, end_time)
        )
    
    def _detect_anomalies(self) -> None:
        """Detect anomalies in hormone levels and system behavior."""
        current_time = datetime.now()
        
        # Check for hormone spikes and drops
        for hormone, level in self.current_hormone_levels.items():
            # Get recent history for comparison
            recent_levels = self._get_recent_hormone_levels(hormone, minutes=5)
            
            if len(recent_levels) > 1:
                avg_level = statistics.mean(recent_levels)
                
                # Check for spike
                if level > avg_level + self.anomaly_thresholds["hormone_spike"]:
                    anomaly = Anomaly(
                        anomaly_type="hormone_spike",
                        component=f"hormone_{hormone}",
                        description=f"Sudden spike in {hormone} levels: {level:.2f} (avg: {avg_level:.2f})",
                        severity="medium",
                        timestamp=current_time.isoformat(),
                        data={"hormone": hormone, "current_level": level, "average_level": avg_level},
                        suggested_actions=[
                            f"Check {hormone} production sources",
                            "Verify cascade triggers",
                            "Monitor for feedback inhibition"
                        ]
                    )
                    self._record_anomaly(anomaly)
                
                # Check for drop
                elif level < avg_level - self.anomaly_thresholds["hormone_drop"]:
                    anomaly = Anomaly(
                        anomaly_type="hormone_drop",
                        component=f"hormone_{hormone}",
                        description=f"Sudden drop in {hormone} levels: {level:.2f} (avg: {avg_level:.2f})",
                        severity="medium",
                        timestamp=current_time.isoformat(),
                        data={"hormone": hormone, "current_level": level, "average_level": avg_level},
                        suggested_actions=[
                            f"Check {hormone} decay rates",
                            "Verify lobe connectivity",
                            "Monitor for excessive consumption"
                        ]
                    )
                    self._record_anomaly(anomaly)
        
        # Check cascade frequency
        recent_cascades = [c for c in self.active_cascades 
                          if (current_time - datetime.fromisoformat(c.timestamp)).total_seconds() < 60]
        
        if len(recent_cascades) > self.anomaly_thresholds["cascade_frequency"]:
            anomaly = Anomaly(
                anomaly_type="excessive_cascades",
                component="hormone_system",
                description=f"Excessive cascade frequency: {len(recent_cascades)} cascades in last minute",
                severity="high",
                timestamp=current_time.isoformat(),
                data={"cascade_count": len(recent_cascades), "cascades": [c.cascade_name for c in recent_cascades]},
                suggested_actions=[
                    "Check cascade trigger thresholds",
                    "Verify feedback inhibition mechanisms",
                    "Monitor for cascade loops"
                ]
            )
            self._record_anomaly(anomaly)
        
        # Check for genetic trigger anomalies
        self.detect_genetic_trigger_anomalies()
    
    def _get_recent_hormone_levels(self, hormone_name: str, minutes: int = 5) -> List[float]:
        """Get recent hormone levels for anomaly detection."""
        start_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT level FROM hormone_levels
                WHERE hormone_name = ? AND timestamp >= ?
                ORDER BY timestamp
            """, (hormone_name, start_time))
            
            return [row[0] for row in cursor.fetchall()]
    
    def _record_anomaly(self, anomaly: Anomaly) -> None:
        """Record an anomaly in the database and notify callbacks."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO anomalies 
                (timestamp, anomaly_type, component, description, severity, data, suggested_actions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                anomaly.timestamp,
                anomaly.anomaly_type,
                anomaly.component,
                anomaly.description,
                anomaly.severity,
                json.dumps(anomaly.data),
                json.dumps(anomaly.suggested_actions)
            ))
            
            conn.commit()
        
        # Notify callbacks
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                self.logger.error(f"Error in anomaly callback: {e}")
        
        self.logger.warning(f"Anomaly detected: {anomaly.description}")
    
    def get_lobe_details(self, lobe_name: str) -> Optional[LobeDetailView]:
        """
        Get detailed information about a specific lobe.
        
        Args:
            lobe_name: Name of the lobe
            
        Returns:
            LobeDetailView object or None if lobe not found
        """
        with self._lock:
            return self.current_lobe_states.get(lobe_name)
    
    def update_lobe_state(self, lobe_detail: LobeDetailView) -> None:
        """
        Update the state of a lobe.
        
        Args:
            lobe_detail: Updated lobe detail view
        """
        with self._lock:
            self.current_lobe_states[lobe_detail.lobe_name] = lobe_detail
        
        self.logger.debug(f"Updated lobe state: {lobe_detail.lobe_name}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent database from growing too large."""
        cutoff_time = (datetime.now() - timedelta(days=7)).isoformat()
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Clean up old hormone levels (keep only recent data)
            cursor.execute("DELETE FROM hormone_levels WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old lobe activity
            cursor.execute("DELETE FROM lobe_activity WHERE timestamp < ?", (cutoff_time,))
            
            # Keep anomalies longer (30 days)
            anomaly_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute("DELETE FROM anomalies WHERE timestamp < ?", (anomaly_cutoff,))
            
            conn.commit()
    
    def register_hormone_callback(self, callback: Callable[[Dict[str, float]], None]) -> None:
        """Register a callback for hormone level updates."""
        self.hormone_callbacks.append(callback)
    
    def register_cascade_callback(self, callback: Callable[[HormoneCascadeEvent], None]) -> None:
        """Register a callback for cascade events."""
        self.cascade_callbacks.append(callback)
    
    def register_anomaly_callback(self, callback: Callable[[Anomaly], None]) -> None:
        """Register a callback for anomaly detection."""
        self.anomaly_callbacks.append(callback)
    
    def register_genetic_trigger_callback(self, callback: Callable[[GeneticTriggerActivation], None]) -> None:
        """Register a callback for genetic trigger activations."""
        self.genetic_trigger_callbacks.append(callback)
    
    def record_genetic_trigger_activation(self, activation: GeneticTriggerActivation) -> None:
        """
        Record a genetic trigger activation event.
        
        Args:
            activation: The genetic trigger activation to record
        """
        with self._lock:
            # Add to active genetic triggers
            self.active_genetic_triggers.append(activation)
            
            # Store in database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO genetic_trigger_activations 
                    (trigger_id, activation_timestamp, environmental_context, activation_score,
                     behavior_changes, performance_impact, duration, success_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    activation.trigger_id,
                    activation.activation_timestamp,
                    json.dumps(activation.environmental_context),
                    activation.activation_score,
                    json.dumps(activation.behavior_changes),
                    json.dumps(activation.performance_impact),
                    activation.duration,
                    json.dumps(activation.success_metrics)
                ))
                
                conn.commit()
        
        # Notify callbacks
        for callback in self.genetic_trigger_callbacks:
            try:
                callback(activation)
            except Exception as e:
                self.logger.error(f"Error in genetic trigger callback: {e}")
        
        self.logger.info(f"Recorded genetic trigger activation: {activation.trigger_id}")
    
    def update_environmental_state(self, env_state: EnvironmentalState) -> None:
        """
        Update the current environmental state.
        
        Args:
            env_state: The current environmental state
        """
        with self._lock:
            self.current_environmental_state = env_state
            
            # Store in database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO environmental_states 
                    (timestamp, system_load, performance_metrics, resource_usage,
                     hormone_levels, task_complexity, adaptation_pressure, network_conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    env_state.timestamp,
                    json.dumps(env_state.system_load),
                    json.dumps(env_state.performance_metrics),
                    json.dumps(env_state.resource_usage),
                    json.dumps(env_state.hormone_levels),
                    env_state.task_complexity,
                    env_state.adaptation_pressure,
                    json.dumps(env_state.network_conditions)
                ))
                
                conn.commit()
        
        self.logger.debug(f"Updated environmental state at {env_state.timestamp}")
    
    def get_genetic_trigger_activations(self, 
                                      trigger_id: Optional[str] = None,
                                      start_time: Optional[str] = None,
                                      end_time: Optional[str] = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get genetic trigger activation history.
        
        Args:
            trigger_id: Optional specific trigger ID to filter by
            start_time: Optional start time (ISO format)
            end_time: Optional end time (ISO format)
            limit: Maximum number of results to return
            
        Returns:
            List of genetic trigger activation records
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT trigger_id, activation_timestamp, environmental_context, 
                       activation_score, behavior_changes, performance_impact, 
                       duration, success_metrics
                FROM genetic_trigger_activations
                WHERE 1=1
            """
            params = []
            
            if trigger_id:
                query += " AND trigger_id = ?"
                params.append(trigger_id)
            
            if start_time:
                query += " AND activation_timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND activation_timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY activation_timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            activations = []
            for row in rows:
                activation = {
                    "trigger_id": row[0],
                    "activation_timestamp": row[1],
                    "environmental_context": json.loads(row[2]) if row[2] else {},
                    "activation_score": row[3],
                    "behavior_changes": json.loads(row[4]) if row[4] else [],
                    "performance_impact": json.loads(row[5]) if row[5] else {},
                    "duration": row[6],
                    "success_metrics": json.loads(row[7]) if row[7] else {}
                }
                activations.append(activation)
            
            return activations
    
    def get_environmental_state_history(self, 
                                      start_time: Optional[str] = None,
                                      end_time: Optional[str] = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get environmental state history.
        
        Args:
            start_time: Optional start time (ISO format)
            end_time: Optional end time (ISO format)
            limit: Maximum number of results to return
            
        Returns:
            List of environmental state records
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT timestamp, system_load, performance_metrics, resource_usage,
                       hormone_levels, task_complexity, adaptation_pressure, network_conditions
                FROM environmental_states
                WHERE 1=1
            """
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            states = []
            for row in rows:
                state = {
                    "timestamp": row[0],
                    "system_load": json.loads(row[1]) if row[1] else {},
                    "performance_metrics": json.loads(row[2]) if row[2] else {},
                    "resource_usage": json.loads(row[3]) if row[3] else {},
                    "hormone_levels": json.loads(row[4]) if row[4] else {},
                    "task_complexity": row[5],
                    "adaptation_pressure": row[6],
                    "network_conditions": json.loads(row[7]) if row[7] else {}
                }
                states.append(state)
            
            return states
    
    def detect_genetic_trigger_anomalies(self) -> None:
        """Detect anomalies in genetic trigger behavior."""
        current_time = datetime.now()
        
        # Check for excessive trigger activations
        recent_activations = [
            t for t in self.active_genetic_triggers 
            if (current_time - datetime.fromisoformat(t.activation_timestamp)).total_seconds() < 300  # 5 minutes
        ]
        
        if len(recent_activations) > 10:  # More than 10 activations in 5 minutes
            anomaly = Anomaly(
                anomaly_type="excessive_genetic_triggers",
                component="genetic_trigger_system",
                description=f"Excessive genetic trigger activations: {len(recent_activations)} in last 5 minutes",
                severity="high",
                timestamp=current_time.isoformat(),
                data={
                    "activation_count": len(recent_activations),
                    "trigger_ids": [t.trigger_id for t in recent_activations]
                },
                suggested_actions=[
                    "Check trigger activation thresholds",
                    "Verify environmental state stability",
                    "Monitor for trigger conflicts"
                ]
            )
            self._record_anomaly(anomaly)
        
        # Check for triggers with consistently low performance impact
        for trigger_id in set(t.trigger_id for t in self.active_genetic_triggers):
            trigger_activations = [t for t in self.active_genetic_triggers if t.trigger_id == trigger_id]
            
            if len(trigger_activations) >= 5:  # Need at least 5 activations to assess
                avg_performance = statistics.mean([
                    sum(t.performance_impact.values()) / len(t.performance_impact) 
                    for t in trigger_activations 
                    if t.performance_impact
                ])
                
                if avg_performance < 0.3:  # Low performance threshold
                    anomaly = Anomaly(
                        anomaly_type="low_performance_genetic_trigger",
                        component=f"genetic_trigger_{trigger_id}",
                        description=f"Genetic trigger {trigger_id} showing consistently low performance: {avg_performance:.2f}",
                        severity="medium",
                        timestamp=current_time.isoformat(),
                        data={
                            "trigger_id": trigger_id,
                            "average_performance": avg_performance,
                            "activation_count": len(trigger_activations)
                        },
                        suggested_actions=[
                            f"Review trigger {trigger_id} configuration",
                            "Consider trigger mutation or replacement",
                            "Analyze environmental context patterns"
                        ]
                    )
                    self._record_anomaly(anomaly)
    
    def generate_genetic_trigger_report(self, 
                                      start_time: Optional[str] = None,
                                      end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report on genetic trigger activity.
        
        Args:
            start_time: Optional start time (ISO format), defaults to 24 hours ago
            end_time: Optional end time (ISO format), defaults to now
            
        Returns:
            Dictionary containing genetic trigger activity report
        """
        if not end_time:
            end_time = datetime.now().isoformat()
        if not start_time:
            start_time = (datetime.now() - timedelta(hours=24)).isoformat()
        
        # Get activations in time range
        activations = self.get_genetic_trigger_activations(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        if not activations:
            return {
                "time_range": (start_time, end_time),
                "total_activations": 0,
                "unique_triggers": 0,
                "summary": "No genetic trigger activations in the specified time range"
            }
        
        # Analyze activations
        trigger_counts = {}
        activation_scores = []
        behavior_changes = []
        performance_impacts = []
        
        for activation in activations:
            trigger_id = activation["trigger_id"]
            trigger_counts[trigger_id] = trigger_counts.get(trigger_id, 0) + 1
            
            activation_scores.append(activation["activation_score"])
            behavior_changes.extend(activation["behavior_changes"])
            
            if activation["performance_impact"]:
                performance_impacts.extend(activation["performance_impact"].values())
        
        # Calculate statistics
        avg_activation_score = statistics.mean(activation_scores) if activation_scores else 0.0
        avg_performance_impact = statistics.mean(performance_impacts) if performance_impacts else 0.0
        
        # Find most active triggers
        most_active_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Count unique behavior changes
        unique_behaviors = list(set(behavior_changes))
        
        return {
            "time_range": (start_time, end_time),
            "total_activations": len(activations),
            "unique_triggers": len(trigger_counts),
            "average_activation_score": avg_activation_score,
            "average_performance_impact": avg_performance_impact,
            "most_active_triggers": most_active_triggers,
            "unique_behavior_changes": len(unique_behaviors),
            "behavior_change_types": unique_behaviors[:10],  # Top 10 behavior types
            "activation_frequency": len(activations) / 24.0,  # Activations per hour
            "trigger_diversity": len(trigger_counts) / max(1, len(activations))  # Diversity ratio
        }
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current monitoring status.
        
        Returns:
            Dictionary with monitoring summary information
        """
        with self._lock:
            return {
                "monitoring_active": self._running,
                "current_hormone_count": len(self.current_hormone_levels),
                "active_cascade_count": len(self.active_cascades),
                "active_genetic_trigger_count": len(self.active_genetic_triggers),
                "monitored_lobe_count": len(self.current_lobe_states),
                "environmental_state_available": self.current_environmental_state is not None,
                "database_path": self.database_path,
                "update_interval": self.update_interval,
                "last_update": datetime.now().isoformat()
            }
    
    def record_performance_metrics(self,
                                 component: str,
                                 implementation: str,
                                 metrics: Dict[str, float],
                                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record performance metrics for a component implementation.
        
        Args:
            component: Component name (e.g., "hormone_calculator")
            implementation: Implementation name (e.g., "algorithmic", "neural")
            metrics: Dictionary of metric names to values
            context: Optional context information
        """
        timestamp = datetime.now().isoformat()
        
        with self._lock:
            # Store in database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                for metric_name, metric_value in metrics.items():
                    cursor.execute("""
                        INSERT INTO performance_metrics 
                        (component, implementation, timestamp, metric_name, metric_value, context)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        component,
                        implementation,
                        timestamp,
                        metric_name,
                        metric_value,
                        json.dumps(context) if context else None
                    ))
                
                conn.commit()
        
        self.logger.debug(f"Recorded performance metrics for {component} ({implementation})")
    
    def record_implementation_switch(self,
                                   component: str,
                                   old_implementation: str,
                                   new_implementation: str,
                                   reason: str,
                                   performance_comparison: Dict[str, Dict[str, float]],
                                   switch_trigger: str = "performance",
                                   confidence_score: float = 0.0,
                                   expected_improvement: Optional[Dict[str, float]] = None) -> None:
        """
        Record an implementation switch event.
        
        Args:
            component: Component name (e.g., "hormone_calculator")
            old_implementation: Previous implementation name
            new_implementation: New implementation name
            reason: Reason for the switch
            performance_comparison: Performance comparison data
            switch_trigger: Trigger for the switch ("performance", "failure", "manual", "scheduled")
            confidence_score: Confidence in the switch decision (0.0 to 1.0)
            expected_improvement: Expected improvement in metrics
        """
        timestamp = datetime.now().isoformat()
        
        with self._lock:
            # Store in database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO implementation_switches 
                    (component, old_implementation, new_implementation, switch_timestamp,
                     reason, performance_comparison, switch_trigger, confidence_score, expected_improvement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    component,
                    old_implementation,
                    new_implementation,
                    timestamp,
                    reason,
                    json.dumps(performance_comparison),
                    switch_trigger,
                    confidence_score,
                    json.dumps(expected_improvement) if expected_improvement else None
                ))
                
                conn.commit()
        
        self.logger.info(f"Recorded implementation switch for {component}: {old_implementation} -> {new_implementation}")
    
    def _serialize_json(self, data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data) if data is not None else None


# Example usage and testing
if __name__ == "__main__":
    # Create monitoring system
    monitoring = MonitoringSystem()
    
    try:
        # Start monitoring
        monitoring.start_monitoring()
        
        # Simulate hormone updates
        monitoring.update_hormone_levels({
            "dopamine": 0.7,
            "serotonin": 0.6,
            "cortisol": 0.3
        }, source_lobe="task_management")
        
        # Simulate cascade event
        cascade = HormoneCascadeEvent(
            cascade_name="reward_cascade",
            trigger_hormone="dopamine",
            trigger_level=0.8,
            affected_hormones=["serotonin", "oxytocin"],
            affected_lobes=["decision_making", "social_intelligence"],
            timestamp=datetime.now().isoformat(),
            duration=2.5,
            effects={"mood_boost": 0.3, "confidence_increase": 0.2}
        )
        monitoring.log_hormone_cascade(cascade)
        
        # Get visualization data
        viz_data = monitoring.generate_time_series_visualization("dopamine")
        print(f"Generated visualization with {len(viz_data.data_points)} data points")
        
        # Get monitoring summary
        summary = monitoring.get_monitoring_summary()
        print(f"Monitoring summary: {summary}")
        
        # Wait a bit to let monitoring run
        time.sleep(3)
        
    finally:
        # Stop monitoring
        monitoring.stop_monitoring()