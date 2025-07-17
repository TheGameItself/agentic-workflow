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
def register_implementation_soring()oniting.stop_m    monitor    onitoring
   # Stop mally:
       fin
      3)
    eep(time.slun
        toring r monito leta bit      # Wait      
   ")
   ations}ndecommeort.rrependations: {f"Recomm  print(     ion}")
 entatlemtive_imp: {report.actionenta implemt(f"Active  prin
      ent}")port.componor {report fformance re perGenerated(f" print      lator")
 ormone_calcu_report("hrisonomparformance_ce_peneratng.ge monitori =     reportort
   nce reperformaate p    # Gener       
  
   )
        }e": 0.5esource_usag5, "rncy": 0.1"late: 0.95, cy""accura metrics={         ral",
  neu"on=tatiimplemen           r",
 alculatone_cmoorent="h     compon     etrics(
  ormance_mord_perftoring.rec        monie metrics
d performanc Recor  #   
        )
   ntve_etch(switchntation_swimpleme_ig.record  monitorin  )
          
  5}": 0.0cyefficien: 0.1, "racy"ccut={"avemenproected_im     exp     =0.8,
  oreence_sc   confid       mance",
  perfor="tch_trigger         swi,
          }.5}
     age": 0rce_us"resou": 0.15, 5, "latency.9uracy": 0": {"acc"neural           0.3},
     ce_usage": urreso"": 0.1, ync.85, "lateracy": 0"accuthmic": {   "algori    {
         ison=arormance_comp        perf",
    r accuracy% betten showed 15ioatplement="Neural imon  reas     ,
     oformat()ow().ise.natetimmestamp=d  switch_ti
          al","neurn=mentatiow_imple         ne
   ic",lgorithmtion="amplementa     old_i      or",
 lculathormone_cant="     compone
       Event(nSwitchatioment= Impleevent   switch_     n switch
 atiomentimpleSimulate      # 
   
        oring()moniting.start_   monitorring
     nitoart mo   # St    
 ry:
    
    tgSystem()Monitorinnitoring = m
    mong systee monitoriat# Cre:
    ""__main__== me__ __nag
if tinand tesge ample usa

# Exanomaly)
aly(ord_anomf._rec   sel                
         )                ]
                s"
    iong decisswitchiniew of revanual der m  "Consi                        hing",
   switcforeection becolle data Increas   "                  ",
       component}for {ria rison critepaance comperformeview   f"R                        ctions=[
  ted_ages       sug        
                 },                dence)
nt_confi": len(receitches"recent_sw                         
   ),nfidenceecent_cotics.mean(ratisdence": stnfiverage_co   "a                       
  mponent,": cocomponent        "                    data={
                        oformat(),
_time.isrrentestamp=cu  tim                   
   ",="medium   severity                f}",
     ence):.2onfid(recent_c.meanistics {statent}: avgomponhes for {citcation swmplementfidence in if"Low concription=es  d                  ",
    ent}componntation_{mplemef"ionent=omp         c          ",
     itchesntation_swpleme_imidence"low_confy_type=alanom                     omaly(
   aly = An  anom         
         ce) < 0.5:nt_confidens.mean(recetictatisdence and s_confi recent       if
                         e > 0]
ence_scorids.conf if s[-5:]witche component_s s inscore force_ [s.confidence =t_confidenen rec            assess
   tches to swi least 3 d atNees) >= 3:  # switchenent_len(compoif                   
  nent]
    ent == compoif s.componches on_switatiment self.impler s ins = [s fowitchent_sompone           ctations:
 _implemenlf.activeent in ser compon fo       scores
  confidencetently lowisth cons wiionsimplementator eck f      # Ch  
  
      (anomaly)cord_anomaly self._re            )
                           ]
            logic"
 omparisontion cy implementaerif      "V            ",
       instabilitymanceforer for p "Check              ,
         omponent}"lds for {chohing threseview switc"R     f                   
ions=[uggested_act       s        },
                    hes]
      s in switc" forion}entatimplem}->{s.new_onplementati_im.old: [f"{shes" "switc                    ,
   es)ch": len(swititch_count"sw                    nent,
    poent": commpon       "co          
       {      data=        
      rmat(),me.isofo_titamp=current   times          
       ="high",rity seve             ,
      t hour"s)} in lasen(switche{lt}: or {componenn switches fntatioplemeExcessive imn=f"riptioesc    d          }",
      ponenttion_{complementanent=f"im     compo              hes",
 tcation_swientive_implemessy_type="exc     anomal             maly(
  Anonomaly =         ar
         in an hou5 switches# More than :  tches) > 5if len(swi        :
    items()hes.witccomponent_s in esnent, switchfor compot
        per componenwitching xcessive sk for eec    # Ch
          
  (switch)endent].app[compont_switches  componen          ]
 [ =omponent]es[ct_switch componen            
   witches: component_sinponent not     if com
        .componentitchnt = sw    compone
        switches:recent_itch in  sw  for     es = {}
 nt_switch     compone
   ntcomponeup by       # Gro      
     ]
   hour
      3600  # 1) <l_seconds()).tota_timestamptchformat(s.swimisoatetime.fro drrent_time - (cu       ifs 
     ion_switchemplementat in self.i   s for s         s = [
nt_switche    recetches
     swientatione implemcessivCheck for ex     #  
      )
    me.now( datetirent_time =ur"
        cavior.""behg itchince and swn performanmentatioes in impleect anomaliet     """D None:
   lf) ->lies(seomatation_anemen_impl detectef  
    d)
        rics
  ficiency_metefetrics=ency_mci        effi   ations,
 endmmdations=recorecommen         nds,
   mance_treperfors=e_trendncperforma         itches,
   istory=swtch_h swi         wn"),
  , "unknoponentget(comns.lementatioctive_impself.antation=emective_impl         a
   l_stats,tations=imp implemen     
       end_time),start_time,e=(   time_rang   (),
      atoformime.now().istetamp=daort_timest      repnt,
      omponemponent=c     co   eport(
    ceRrformann Pe   retur  
     
      t_sumghore / weificiency_scefs[impl] = riciciency_met         eff      :
      > 0eight_sum      if w         
                 t
sum += weight_       weigh            ized
     mal* noreight e += wscorfficiency_    e                             
            etter
   lower is bert so ue))  # Inv, val(1.0min1.0 - = max(0.0, lized orma          n         
         e_usagey, resourc:  # latenc        else              
  tage percen 0-1 oready alralues aresume vlue)  # As0, va= min(1.ormalized      n                   :
    "]hput "throug"accuracy",in [e _nam if metric                     fied)
  s simplis icale (thi1 slize to 0-     # Norma                      
            ]
         me]["mean"naetric_tats[malue = s      v        
           stats:_name in  if metric              :
    .items()tric_weightseight in meme, wr metric_na         fo      
                }
                ": 0.1
 throughput          "       
   0.2,age": source_us "re             .3,
      ncy": 0   "late           .4,
      ": 0"accuracy                    s = {
ight   metric_we        rics
     rent metiffe  # Weight d    
                         sum = 0.0
     weight_           ore = 0.0
 iciency_scff         e
       .items():mpl_statss in impl, statfor i           stats:
 f impl_    i   {}
  s =iency_metric effic      
  metricsefficiencylate Calcu      #        
  ")
 0):.1f}%vg * 10der_ag) / ol older_avecent_avg -(rd by {(s worsenee} ha {metric_nammpl}pend(f"{idations.apmmen  reco                
          g * 1.1:older_avecent_avg >      if r                etter)
   wer is blource_usage (soor_rate, rerratency, eelse:  # l                   
 0):.1f}%")10_avg * er old_avg) /enter_avg - rec{((oldraded by me} has deg_na {metricmpl}end(f"{idations.appmmen      reco                
      avg * 0.9:lder_t_avg < o    if recen                  
  te"]:ess_raut", "succoughp "thrfficiency", "e",accuracy in ["me.lower()nametric_         if           
 rse)10% woion (>gradatficant denick for sig Che        #            
             
       avg recent_ 5 else(values) >]) if lens[:-5s.mean(valuesticavg = statiolder_            ])
        ues[-5:an(valatistics.me_avg = stecent         r          ints
 poata gh denou 5:  # Need alues) >=    if len(v    :
        s.items()n trend, values itric_name  for me     ms():
     tends.iformance_tren per i, trendsimpl    for    ion
 radategrformance dpe Check for       #     
  est")
   performs bn tatiolement_impl} impname}, {besic_etr(f"For {mons.appendmendati recom            pl:
         if best_im                
   
           uee = val   best_valu                          pl
   impl = im  best_                       
       best_value:<  value e ore is Nonbest_valu    if                   
      urce_usagesote, rey, error_ra:  # latenc       else                 e
e = valuvalu    best_                  
          l= imppl imest_         b               e:
        best_valu > aluer vis None oe valu  if best_                  
        e"]:success_rat", ghput"ou", "thriciencyeff", "ccuracyin ["awer() c_name.lof metri   i                   ate
  or_rlatency, errer for er is bettncy; lowy, efficieurac acctter for is besume higherAs        #                   
                "]
      anname]["memetric_ts[ value = sta                
       n stats:_name imetric        if      
       tems():_stats.its in implmpl, sta    for i               
            None
 = ue   best_val             one
 l = N best_imp           
    alues()]):stats.vpl_ims in tricimpl_me.keys() for metricsnion(*[impl_ in set().uic_namer metr        fo    c
etriach mation for ementle impt performing# Find bes        
     1:stats) >mpl_  if len(ie
      ve multiplf we haentations iimplem Compare         #    
= []
    tions   recommendans
      endatio recommrate# Gene    
            }
           
         n(values)let":      "coun           0,
        e 0.es) > 1 els len(valu ifs).stdev(valuecs": statisti   "std_dev               ),
      valuesx( "max": ma             
          ),(valuesmin": min          "              lues),
dian(vatics.metatisedian": s "m               
        ean(values),tatistics.m sn": "mea                    = {
   name] mpl][metric_mpl_stats[i           i         s:
 value   if            ems():
 itmetrics.in impl_alues tric_name, vfor me        = {}
     impl]pl_stats[   im        ems():
 s.itionentat in implemricsmpl_metl, ior imp     f  ts = {}
 impl_sta  ion
      ntatpleme for each imatisticsalculate st    # C      
    
  e)_valumetricnd(ppe].aetric_namends[impl][mance_tre     perform     lue)
  ric_vaappend(metame].ric_nmpl][metentations[iimplem  
                     = []
  me]][metric_nas[impl_trendrmance      perfo
          _name] = []mpl][metrictations[ilemenmp i       
        ions[impl]:atementmplot in ietric_name nf m   i            
    {}
      ends[impl] =rmance_tr     perfo    
       pl] = {}[imementations      impl    
      entations:leml not in imp   if imp            
        alue"]
 "metric_vmetric[lue = metric_va         name"]
   c_ic["metri_name = metr  metric
          entation"]"implemtric[l = me       impics:
     tric in metr for me  
          s = {}
   _trend performance       ions = {}
lementat    imp  ntation
  lemes by impze metricOrgani       # 
        )
       me
  nd_ti end_time=e      
     me,t_time=startiart_     st     
  t=component,omponen    c
        switches(ementation_et_impl= self.ges witch sory
       isttch h# Get swi              
  )
 
       d_time end_time=en           t_time,
arime=st    start_t
        nt,=componeent     compon       trics(
mance_meperforelf.get_ metrics = s       
me range the tior fcsnce metrirmat perfo        # Ge 
      ormat()
 4)).isofrs=2a(houeltw() - timedatetime.nort_time = (d        sta   
 _time:artnot st  if )
      ).isoformat(ime.now(e = datetim    end_t  me:
      t end_ti no
        if"        ""is
nsive analysrehempith coect w objeportnceR  Performa
          s:eturn       R    
        
 to nowults ), defamate (ISO fornd timonal e_time: Opti       end
     ours ago to 24 h defaultsO format),IS ( timeartptional stme: O   start_ti
         ntomponethe cof t: Name enmpon co     
      rgs:        A       
mponent.
 rt for a con repoe comparisomancnsive perfore a comprehe  Generat      "
""   t:
     manceRepor> Perforone) -tr] = Nptional[sme: O end_ti                                           e,
 [str] = Non Optionalime:rt_t   sta                                      : str,
    component                                     
        f, selon_report(ismpare_coperformancrate_   def gene   
 rics
   return met               
  
     etric).append(mics metr          
     }              {}
  e ] els if row[4(row[4]): json.loadsext"    "cont     
           : row[3],ue"almetric_v "                  
 ow[2],c_name": rtri      "me      ],
        amp": row[1   "timest        
         : row[0],tation"emen    "impl         {
        tric =        mes:
        rowin    for row 
         ics = []    metr       onaries
  to dicti# Convert          
          ()
    fetchallcursor.     rows =   ams)
     (query, parrsor.execute cu            
        
   pend(limit)ms.ap para          MIT ?"
 p DESC LIamimestBY tER " ORDquery +=                  
   )
    timed_end(enrams.apppa       
         <= ?"amp timesty += " AND uer     q    
       f end_time:        i      
       me)
   _tipend(start params.ap       "
        ?tamp >= " AND times  +=   query       :
      mert_ti sta    if         
     
      n)mentatioimpleend(rams.app       pa      "
   = ?ation ntemempl+= " AND iry      que     n:
      ntatioeme  if impl        
             
 component]  params = [        
       """= ?
       E component     WHER          metrics
  ce_rformanM pe FRO               t
 contexic_value,ame, metrmetric_nestamp, ntation, tim impleme    SELECT            ""
query = "     
                  rsor()
 conn.cu   cursor =     n:
     th) as cone_palf.databast(se.connece3ith sqlit  w   ""
   "      records
  e metric formancList of per         s:
   rn   Retu            
 urn
     retf results tor omum numbet: Maxi        limi
    rmat)ime (ISO fol end tona: Optid_time     en      
 SO format) time (Ial startontitime: Opart_      st     lter by
 to fion mplementatiecific i Optional spn:mentatio imple       
    onent comptheof Name t: omponen          c   Args:
          
     ry.
 histoetricsmance morrfet pe  G   "
        ""   tr, Any]]:
[Dict[s) -> List 1000: int =   limit                           ne,
r] = Noal[sttionme: Op  end_ti                         None,
   = ] stre: Optional[tart_tim         s                   e,
  = Nonional[str] tion: Optementa    impl                    str,
      nent:  compo                       f, 
      el(snce_metricst_performaef ge    d
    
switchesturn    re             
    
    switch)end(appches.     swit          }
             {}
    ow[8] else if rrow[8]) on.loads(t": js_improvemen"expected                     row[7],
_score":"confidence                   ],
 ger": row[6ig "switch_tr                {},
   ] else  row[5w[5]) ifads(rolon.n": jsoparisormance_com  "perfo           
       : row[4],"reason"                    : row[3],
amp"estim  "switch_t               row[2],
   ion": mplementat  "new_i                 
 : row[1],on"ementatipl   "old_im                  row[0],
onent":      "comp           {
      switch =            s:
  rowrow in or            f[]
 switches =       ies
      art to diction # Conver         
            hall()
  fetcsor.rows = cur     )
       y, paramsxecute(querr.ecurso    
                   imit)
 ms.append(lara   p    
      LIMIT ?"stamp DESCitch_timesw ORDER BY y += "      quer            
    _time)
  nd(endrams.appe   pa         ?"
     p <=ch_timestam" AND switquery +=               :
  me  if end_ti
                      me)
rt_ti(sta.appendams   par          
   ?"stamp >= _timechD swit" AN += ry   que            :
 tart_time if s               
  )
      ntd(componeams.appen       par
         "ponent = ?omAND c" =     query +            omponent:
 c         if     
   
       params = []     
         """     1=1
     E     WHER    es
        switchementation_FROM impl      
          ntmeted_improvecore, expecence_s, confidgerwitch_trigomparison, se_cmanc perforeason,      r        
         amp,witch_timestion, splementattion, new_imenta_implemnt, oldnempoT coELEC    S          ""
  = " query     
                   ursor()
onn.c= c cursor 
           ) as conn:athabase_pf.datelonnect(ssqlite3.cwith "
        ""    
    switch recordtation slemen List of imp           turns:
  Re    
        rn
       retuults to resofnumber Maximum  limit:           ormat)
  time (ISO fl end: Optiona    end_time
        mat)ime (ISO forstart tonal ptitart_time: O  s
          er bynent to filtompo cal specific: Optionentcompon              Args:

              .
ch historyitentation swplemt im        Ge"
      ""
   Any]]:ict[str,) -> List[D = 100limit: int                               None,
   r] = Optional[st:     end_time                        ne,
      str] = No Optional[start_time:                               None,
    nal[str] =nt: Optio  compone                           
      itches(self,tion_swimplementa def get_)
    
    {metrics}"entation}:}:{implemntompone for {cnce metrics performaordedebug(f"Reclf.logger.d
        se      
  ()mit    conn.com 
             
      ))t else None contexntext) ifn.dumps(co   jso                lue,
  vaic_ame, metr_n, metric timestampentation,nt, implemnempo (co    """,            ?, ?, ?)
, (?, ?, ?VALUES                  t)
   , contexuetric_valname, memp, metric_staon, timeati, implementmponent (co             
       csance_metriINTO performT SER     IN              "
 e(""ursor.execut c               ems():
etrics.itvalue in mmetric_etric_name,    for m
                    
 n.cursor()cursor = con             conn:
se_path) asself.databa3.connect(qlite     with s  
   
      soformat().i.now()tetimedamp = timesta"
        "        "ion
ormattext infptional con Otext:        conalues
    es to vric nametary of mionctetrics: Di    m     ic")
   hm"algorit", ural(e.g., "neentation of implemtion: Type ta  implemen
          e componente of th: Namponentom         c  Args:
   
           tion.
    implementa specificr ae metrics fod performanc  Recor      """
       
  None:= None) ->] str, Any]nal[Dict[t: Optio contex                           oat],
     t[str, flicmetrics: D                          
       : str,ntation impleme                       
         str,omponent:    c                            (self, 
  e_metricsrformancrd_pe    def reco
    
ation}")_implementvent.new_ewitch {s} tolementationold_impch_event.from {swit        f"                onent} "
ent.comph_evtcch: {swiation switplement imecorded"Rgger.info(f.lo        self        
 {e}")
ck:bach callittion swlementarror in impor(f"Ef.logger.err   sel            on as e:
 cept Excepti ex         
  event)ck(switch_baall           cry:
                tallbacks:
 ch_cion_swit.implementatin self callback   for     
 lbackscalfy      # Noti
    )
       nn.commit(   co             
             
           ))    ent)
    rovemd_impexpectech_event.umps(switson.d       j          
   nce_score,nt.confideve switch_e                  rigger,
 witch_th_event.stc    swi               ),
 comparisonformance_h_event.perumps(switc    json.d            n,
    _event.reaso switch              tamp,
     itch_timesevent.sw     switch_       
        ation,entew_implemch_event.n      swit           tation,
   lemenld_imptch_event.oswi                    
mponent,tch_event.co     swi              """, (
           
       ?, ?), ?, ?,?, ?, ?, ?,   VALUES (?                ent)
  rovemcted_imp_score, expe confidencer,tch_triggeparison, swirmance_com, perfoason       re            ,
  timestamp, switch_plementationon, new_imtatimenold_imple, entonmp(co             s 
       ion_switcheat implementINSERT INTO                
    te("""rsor.execu       cu
                )
         cursor(nn.r = co       curso         
 conn:path) astabase_f.daelconnect(sith sqlite3.     w    ase
   atabe in d# Stor           
           tion
  ementat.new_impl_evenwitchmponent] = sent.coch_evitions[sw_implementatlf.active         se
   tationve implemenctidate a    # Up        
            ch_event)
wit(sches.appenditon_swplementati     self.im
       n switchesmplementatioto i# Add            
 ock:f._lth sel     wi    """
       record
 toh event itcion sw implementatt: Thewitch_even        s Args:
                   
tch event.
swientation emimpl Record an    
    "    ""   e:
  Nonnt) ->nSwitchEventatiompleme: Ivent_echself, switswitch(tation_rd_implemen def reco)
    
   llbackcas.append(ckitch_callbaion_swementatmpl      self.i."""
  ch eventsntation switfor implemeback ter a call""Regis       "> None:
 ) -one]vent], NationSwitchEe[[Implementablk: Calllbac callf,callback(sewitch_