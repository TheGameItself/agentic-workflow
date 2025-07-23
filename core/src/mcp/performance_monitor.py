#!/usr/bin/env python3
"""
Enhanced Performance Monitor for MCP Core System
Comprehensive performance monitoring with brain-inspired metrics and optimization.
"""

import asyncio
import logging
import psutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union
import json
import statistics

class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert information."""
    metric_name: str
    level: AlertLevel
    message: str
    value: Union[int, float]
    threshold: Union[int, float]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class MetricCollector:
    """Collects and manages individual metrics."""
    
    def __init__(self, name: str, metric_type: MetricType, max_history: int = 1000):
        self.name = name
        self.type = metric_type
        self.max_history = max_history
        self.values = deque(maxlen=max_history)
        self.lock = threading.Lock()
        
        # Aggregated statistics
        self._sum = 0.0
        self._count = 0
        self._min = float('inf')
        self._max = float('-inf')
    
    def record(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None, 
               metadata: Optional[Dict[str, Any]] = None):
        """Record a new metric value."""
        with self.lock:
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            self.values.append(metric_value)
            
            # Update aggregated statistics
            self._sum += value
            self._count += 1
            self._min = min(self._min, value)
            self._max = max(self._max, value)
    
    def get_current_value(self) -> Optional[Union[int, float]]:
        """Get the most recent metric value."""
        with self.lock:
            return self.values[-1].value if self.values else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of the metric."""
        with self.lock:
            if not self.values:
                return {}
            
            values = [v.value for v in self.values]
            
            return {
                'count': len(values),
                'sum': self._sum,
                'min': self._min if self._min != float('inf') else None,
                'max': self._max if self._max != float('-inf') else None,
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'percentile_95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else None,
                'percentile_99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else None,
                'current': self.get_current_value(),
                'last_updated': self.values[-1].timestamp.isoformat() if self.values else None
            }
    
    def get_values_in_range(self, start_time: datetime, end_time: datetime) -> List[MetricValue]:
        """Get metric values within a time range."""
        with self.lock:
            return [
                v for v in self.values
                if start_time <= v.timestamp <= end_time
            ]

class ObjectivePerformanceMonitor:
    """
    Enhanced performance monitor with brain-inspired metrics.
    
    Features:
    - Multi-dimensional metric collection
    - Real-time alerting system
    - Performance trend analysis
    - Resource optimization recommendations
    - Brain-inspired hormone system integration
    - Automatic performance tuning
    """
    
    def __init__(self, enable_system_metrics: bool = True, 
                 enable_alerts: bool = True, 
                 alert_callback: Optional[Callable] = None):
        self.enable_system_metrics = enable_system_metrics
        self.enable_alerts = enable_alerts
        self.alert_callback = alert_callback
        
        # Core components
        self.metrics: Dict[str, MetricCollector] = {}
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds: Dict[str, Dict[str, Union[int, float]]] = {}
        
        # System monitoring
        self.system_monitor_task = None
        self.alert_monitor_task = None
        self.optimization_task = None
        
        # Thread safety
        self.lock = threading.RLock()
        self.running = False
        
        # Performance optimization state
        self.optimization_history = deque(maxlen=100)
        self.performance_baselines = {}
        
        # Brain-inspired components
        self.hormone_levels = {
            'stress': 0.0,      # High when system under load
            'efficiency': 1.0,   # High when system performing well
            'adaptation': 0.5,   # Drives system changes
            'stability': 1.0     # High when system is stable
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default metrics and thresholds
        self._initialize_default_metrics()
        self._initialize_alert_thresholds()
        
        self.logger.info("Enhanced Performance Monitor initialized")
    
    def _initialize_default_metrics(self):
        """Initialize default system metrics."""
        default_metrics = [
            ('cpu_usage', MetricType.GAUGE),
            ('memory_usage', MetricType.GAUGE),
            ('disk_usage', MetricType.GAUGE),
            ('network_io', MetricType.COUNTER),
            ('response_time', MetricType.TIMER),
            ('request_count', MetricType.COUNTER),
            ('error_count', MetricType.COUNTER),
            ('active_connections', MetricType.GAUGE),
            ('queue_size', MetricType.GAUGE),
            ('throughput', MetricType.GAUGE),
            
            # Brain-inspired metrics
            ('cognitive_load', MetricType.GAUGE),
            ('attention_focus', MetricType.GAUGE),
            ('memory_consolidation', MetricType.GAUGE),
            ('learning_rate', MetricType.GAUGE),
            ('adaptation_speed', MetricType.GAUGE),
            ('system_coherence', MetricType.GAUGE),
        ]
        
        for name, metric_type in default_metrics:
            self.create_metric(name, metric_type)
    
    def _initialize_alert_thresholds(self):
        """Initialize default alert thresholds."""
        self.alert_thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'disk_usage': {'warning': 90.0, 'critical': 98.0},
            'response_time': {'warning': 1.0, 'critical': 5.0},
            'error_count': {'warning': 10, 'critical': 50},
            'cognitive_load': {'warning': 0.8, 'critical': 0.95},
            'system_coherence': {'warning': 0.3, 'critical': 0.1},  # Low values are bad
        }
    
    def create_metric(self, name: str, metric_type: MetricType, max_history: int = 1000) -> bool:
        """Create a new metric collector."""
        try:
            with self.lock:
                if name not in self.metrics:
                    self.metrics[name] = MetricCollector(name, metric_type, max_history)
                    self.logger.debug(f"Created metric: {name} ({metric_type.value})")
                    return True
                else:
                    self.logger.warning(f"Metric {name} already exists")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to create metric {name}: {e}")
            return False
    
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value."""
        try:
            with self.lock:
                if name not in self.metrics:
                    # Auto-create gauge metric if it doesn't exist
                    self.create_metric(name, MetricType.GAUGE)
                
                self.metrics[name].record(value, labels, metadata)
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_alert_conditions(name, value)
                
                # Update hormone levels based on key metrics
                self._update_hormone_levels(name, value)
                
        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
    
    def increment_counter(self, name: str, increment: Union[int, float] = 1,
                         labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        current_value = self.get_current_value(name) or 0
        self.record_metric(name, current_value + increment, labels)
    
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return TimerContext(self, name)
    
    def get_current_value(self, name: str) -> Optional[Union[int, float]]:
        """Get the current value of a metric."""
        with self.lock:
            if name in self.metrics:
                return self.metrics[name].get_current_value()
        return None
    
    def get_metric_statistics(self, name: str) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        with self.lock:
            if name in self.metrics:
                return self.metrics[name].get_statistics()
        return {}
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics."""
        with self.lock:
            return {
                name: collector.get_statistics()
                for name, collector in self.metrics.items()
            }
    
    def set_alert_threshold(self, metric_name: str, level: str, threshold: Union[int, float]):
        """Set alert threshold for a metric."""
        with self.lock:
            if metric_name not in self.alert_thresholds:
                self.alert_thresholds[metric_name] = {}
            self.alert_thresholds[metric_name][level] = threshold
    
    def _check_alert_conditions(self, metric_name: str, value: Union[int, float]):
        """Check if metric value triggers any alerts."""
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        
        # Check critical threshold first
        if 'critical' in thresholds and value >= thresholds['critical']:
            self._trigger_alert(metric_name, AlertLevel.CRITICAL, value, thresholds['critical'])
        elif 'warning' in thresholds and value >= thresholds['warning']:
            self._trigger_alert(metric_name, AlertLevel.WARNING, value, thresholds['warning'])
    
    def _trigger_alert(self, metric_name: str, level: AlertLevel, 
                      value: Union[int, float], threshold: Union[int, float]):
        """Trigger a performance alert."""
        alert = PerformanceAlert(
            metric_name=metric_name,
            level=level,
            message=f"{metric_name} is {value}, exceeding {level.value} threshold of {threshold}",
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        with self.lock:
            self.alerts.append(alert)
            
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        self.logger.warning(f"Performance alert: {alert.message}")
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _update_hormone_levels(self, metric_name: str, value: Union[int, float]):
        """Update brain-inspired hormone levels based on metrics."""
        try:
            # Stress hormone - increases with high resource usage
            if metric_name in ['cpu_usage', 'memory_usage', 'response_time']:
                if metric_name == 'response_time':
                    stress_factor = min(value / 5.0, 1.0)  # Normalize to 0-1
                else:
                    stress_factor = value / 100.0  # Percentage to 0-1
                
                self.hormone_levels['stress'] = max(
                    self.hormone_levels['stress'] * 0.9 + stress_factor * 0.1,
                    0.0
                )
            
            # Efficiency hormone - decreases with errors and high response times
            if metric_name in ['error_count', 'response_time']:
                if metric_name == 'error_count':
                    efficiency_factor = max(1.0 - value / 100.0, 0.0)
                else:
                    efficiency_factor = max(1.0 - value / 10.0, 0.0)
                
                self.hormone_levels['efficiency'] = (
                    self.hormone_levels['efficiency'] * 0.95 + efficiency_factor * 0.05
                )
            
            # Adaptation hormone - increases when performance changes
            if metric_name in ['throughput', 'response_time']:
                baseline = self.performance_baselines.get(metric_name, value)
                change_ratio = abs(value - baseline) / max(baseline, 1.0)
                adaptation_factor = min(change_ratio, 1.0)
                
                self.hormone_levels['adaptation'] = (
                    self.hormone_levels['adaptation'] * 0.8 + adaptation_factor * 0.2
                )
                
                # Update baseline
                self.performance_baselines[metric_name] = (
                    baseline * 0.99 + value * 0.01
                )
            
            # Stability hormone - decreases with high variability
            if metric_name in self.metrics:
                stats = self.metrics[metric_name].get_statistics()
                if stats.get('stdev') is not None and stats.get('mean') is not None:
                    cv = stats['stdev'] / max(stats['mean'], 1.0)  # Coefficient of variation
                    stability_factor = max(1.0 - cv, 0.0)
                    
                    self.hormone_levels['stability'] = (
                        self.hormone_levels['stability'] * 0.9 + stability_factor * 0.1
                    )
            
            # Clamp hormone levels to valid range
            for hormone in self.hormone_levels:
                self.hormone_levels[hormone] = max(0.0, min(1.0, self.hormone_levels[hormone]))
                
        except Exception as e:
            self.logger.error(f"Failed to update hormone levels: {e}")
    
    def get_hormone_levels(self) -> Dict[str, float]:
        """Get current brain-inspired hormone levels."""
        return self.hormone_levels.copy()
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        try:
            # Weight different factors
            weights = {
                'efficiency': 0.3,
                'stability': 0.25,
                'stress': -0.25,  # Negative because high stress is bad
                'adaptation': 0.2
            }
            
            score = 0.0
            for hormone, weight in weights.items():
                if weight < 0:
                    score += (1.0 - self.hormone_levels[hormone]) * abs(weight)
                else:
                    score += self.hormone_levels[hormone] * weight
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate health score: {e}")
            return 0.5  # Default neutral score
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        try:
            # Check hormone levels for recommendations
            if self.hormone_levels['stress'] > 0.8:
                recommendations.append("High system stress detected. Consider scaling resources or optimizing workload.")
            
            if self.hormone_levels['efficiency'] < 0.3:
                recommendations.append("Low efficiency detected. Review error rates and response times.")
            
            if self.hormone_levels['stability'] < 0.4:
                recommendations.append("System instability detected. Check for resource contention or configuration issues.")
            
            if self.hormone_levels['adaptation'] > 0.7:
                recommendations.append("High adaptation activity. System may be responding to changing conditions.")
            
            # Check specific metrics
            cpu_usage = self.get_current_value('cpu_usage')
            if cpu_usage and cpu_usage > 90:
                recommendations.append("CPU usage is very high. Consider optimizing CPU-intensive operations.")
            
            memory_usage = self.get_current_value('memory_usage')
            if memory_usage and memory_usage > 90:
                recommendations.append("Memory usage is very high. Consider memory optimization or garbage collection.")
            
            response_time = self.get_current_value('response_time')
            if response_time and response_time > 2.0:
                recommendations.append("Response times are high. Consider caching, indexing, or query optimization.")
            
            error_count = self.get_current_value('error_count')
            if error_count and error_count > 20:
                recommendations.append("High error count detected. Review error logs and fix underlying issues.")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to monitoring error.")
        
        return recommendations
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self.running:
            return
        
        self.running = True
        
        if self.enable_system_metrics:
            self.system_monitor_task = asyncio.create_task(self._system_monitor_loop())
        
        if self.enable_alerts:
            self.alert_monitor_task = asyncio.create_task(self._alert_monitor_loop())
        
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self.running = False
        
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
        
        if self.alert_monitor_task:
            self.alert_monitor_task.cancel()
        
        if self.optimization_task:
            self.optimization_task.cancel()
        
        self.logger.info("Performance monitoring stopped")
    
    async def _system_monitor_loop(self):
        """Background system monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                self.record_metric('cpu_usage', psutil.cpu_percent())
                self.record_metric('memory_usage', psutil.virtual_memory().percent)
                self.record_metric('disk_usage', psutil.disk_usage('/').percent)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.record_metric('network_io', net_io.bytes_sent + net_io.bytes_recv)
                
                # Calculate cognitive load (simplified)
                cpu = self.get_current_value('cpu_usage') or 0
                memory = self.get_current_value('memory_usage') or 0
                cognitive_load = (cpu + memory) / 200.0  # Normalize to 0-1
                self.record_metric('cognitive_load', cognitive_load)
                
                # Calculate system coherence
                health_score = self.get_system_health_score()
                self.record_metric('system_coherence', health_score)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _alert_monitor_loop(self):
        """Background alert monitoring loop."""
        while self.running:
            try:
                # Check for alert resolution
                current_time = datetime.now()
                
                with self.lock:
                    for alert in self.alerts:
                        if not alert.resolved:
                            current_value = self.get_current_value(alert.metric_name)
                            if current_value is not None and current_value < alert.threshold:
                                alert.resolved = True
                                alert.resolution_time = current_time
                                self.logger.info(f"Alert resolved: {alert.message}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while self.running:
            try:
                # Perform optimization analysis
                recommendations = self.get_performance_recommendations()
                
                if recommendations:
                    self.logger.info(f"Performance recommendations: {recommendations}")
                
                # Record optimization metrics
                health_score = self.get_system_health_score()
                self.optimization_history.append({
                    'timestamp': datetime.now(),
                    'health_score': health_score,
                    'hormone_levels': self.hormone_levels.copy(),
                    'recommendations': recommendations
                })
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export all metrics in specified format."""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.get_all_metrics(),
                'hormone_levels': self.get_hormone_levels(),
                'health_score': self.get_system_health_score(),
                'alerts': [
                    {
                        'metric_name': alert.metric_name,
                        'level': alert.level.value,
                        'message': alert.message,
                        'value': alert.value,
                        'threshold': alert.threshold,
                        'timestamp': alert.timestamp.isoformat(),
                        'resolved': alert.resolved
                    }
                    for alert in self.alerts[-50:]  # Last 50 alerts
                ]
            }
            
            if format_type.lower() == 'json':
                return json.dumps(data, indent=2)
            else:
                return str(data)
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return f"Export failed: {e}"

class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: ObjectivePerformanceMonitor, metric_name: str):
        self.monitor = monitor
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_metric(self.metric_name, duration)

# Convenience functions

def create_performance_monitor(**kwargs) -> ObjectivePerformanceMonitor:
    """Create a performance monitor with custom settings."""
    return ObjectivePerformanceMonitor(**kwargs)

def get_system_performance_summary() -> Dict[str, Any]:
    """Get a quick system performance summary."""
    try:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}