"""
Enhanced Monitoring and Visualization System

This module provides comprehensive real-time monitoring and visualization
of the MCP system's brain-inspired architecture, including hormone levels,
neural network performance, genetic triggers, and system-wide metrics.

Features:
- Real-time hormone level visualization
- Neural vs algorithmic performance comparison
- Genetic trigger activation tracking
- System-wide performance metrics
- Anomaly detection and alerting
- Historical trend analysis
- Interactive dashboard data generation
"""

import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import threading

from .hormone_system_controller import HormoneSystemController
from .neural_network_models.hormone_neural_integration import HormoneNeuralIntegration
from .genetic_trigger_system.integrated_genetic_system import IntegratedGeneticTriggerSystem
from .brain_state_aggregator import BrainStateAggregator


@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: datetime
    hormone_levels: Dict[str, float]
    neural_performance: Dict[str, Dict[str, float]]
    genetic_triggers: Dict[str, Any]
    resource_usage: Dict[str, float]
    lobe_states: Dict[str, Dict[str, Any]]
    performance_scores: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    system_health: float


@dataclass
class VisualizationData:
    """Data structure for visualization components"""
    chart_type: str
    data: Dict[str, Any]
    options: Dict[str, Any]
    timestamp: datetime


@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]
    recommendations: List[str]


class EnhancedMonitoringSystem:
    """
    Enhanced monitoring system for comprehensive MCP system tracking.
    
    Provides real-time monitoring, visualization data generation,
    anomaly detection, and performance analysis for all system components.
    """
    
    def __init__(self, 
                 hormone_system: Optional[HormoneSystemController] = None,
                 neural_integration: Optional[HormoneNeuralIntegration] = None,
                 genetic_system: Optional[IntegratedGeneticTriggerSystem] = None,
                 brain_state: Optional[BrainStateAggregator] = None):
        
        self.logger = logging.getLogger("EnhancedMonitoring")
        
        # System components
        self.hormone_system = hormone_system
        self.neural_integration = neural_integration
        self.genetic_system = genetic_system
        self.brain_state = brain_state
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.anomaly_history: List[AnomalyAlert] = []
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Real-time tracking
        self.current_metrics: Optional[SystemMetrics] = None
        self.active_alerts: Set[str] = set()
        
        # Configuration
        self.monitoring_interval = 1.0  # seconds
        self.anomaly_thresholds = {
            'hormone_spike': 0.8,
            'performance_drop': 0.3,
            'resource_high': 0.9,
            'neural_failure': 0.1
        }
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Callbacks for external systems
        self.alert_callbacks: List[callable] = []
        self.visualization_callbacks: List[callable] = []
        
        self.logger.info("Enhanced Monitoring System initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Continuous monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Continuous monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_system_metrics()
                self.current_metrics = metrics
                
                # Store in history
                self.metrics_history.append(metrics)
                
                # Check for anomalies
                anomalies = self._detect_anomalies(metrics)
                if anomalies:
                    self._handle_anomalies(anomalies)
                
                # Update performance history
                self._update_performance_history(metrics)
                
                # Generate visualization data
                viz_data = self._generate_visualization_data()
                
                # Trigger callbacks
                self._trigger_callbacks(metrics, anomalies, viz_data)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        timestamp = datetime.now()
        
        # Collect hormone levels
        hormone_levels = {}
        if self.hormone_system:
            hormone_levels = self.hormone_system.get_levels()
        
        # Collect neural performance
        neural_performance = {}
        if self.neural_integration:
            status = self.neural_integration.get_system_status()
            neural_performance = {
                'implementations': status.get('current_implementations', {}),
                'performance_summary': status.get('performance_summary', {}),
                'models_available': status.get('neural_models_available', 0)
            }
        
        # Collect genetic trigger data
        genetic_triggers = {}
        if self.genetic_system:
            genetic_triggers = {
                'active_triggers': list(self.genetic_system.active_triggers),
                'total_triggers': len(self.genetic_system.triggers),
                'generation': self.genetic_system.generation,
                'success_rate': self.genetic_system.successful_activations / max(1, self.genetic_system.total_activations)
            }
        
        # Collect resource usage
        resource_usage = self._collect_resource_usage()
        
        # Collect lobe states
        lobe_states = {}
        if self.brain_state:
            lobe_states = self._collect_lobe_states()
        
        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(
            hormone_levels, neural_performance, genetic_triggers, resource_usage
        )
        
        # Detect anomalies
        anomalies = self._detect_anomalies_in_metrics(
            hormone_levels, neural_performance, genetic_triggers, resource_usage
        )
        
        # Calculate system health
        system_health = self._calculate_system_health(performance_scores, anomalies)
        
        return SystemMetrics(
            timestamp=timestamp,
            hormone_levels=hormone_levels,
            neural_performance=neural_performance,
            genetic_triggers=genetic_triggers,
            resource_usage=resource_usage,
            lobe_states=lobe_states,
            performance_scores=performance_scores,
            anomalies=anomalies,
            system_health=system_health
        )
    
    def _collect_resource_usage(self) -> Dict[str, float]:
        """Collect system resource usage"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            }
        except ImportError:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_percent': 0.0,
                'network_io': 0.0
            }
    
    def _collect_lobe_states(self) -> Dict[str, Dict[str, Any]]:
        """Collect lobe state information"""
        lobe_states = {}
        
        if self.brain_state and hasattr(self.brain_state, 'lobes'):
            for lobe_name, lobe in self.brain_state.lobes.items():
                lobe_states[lobe_name] = {
                    'active': getattr(lobe, 'active', False),
                    'hormone_levels': getattr(lobe, 'local_hormone_levels', {}),
                    'performance': getattr(lobe, 'performance_metrics', {}),
                    'last_activity': getattr(lobe, 'last_activity', None)
                }
        
        return lobe_states
    
    def _calculate_performance_scores(self, hormone_levels: Dict[str, float],
                                    neural_performance: Dict[str, Any],
                                    genetic_triggers: Dict[str, Any],
                                    resource_usage: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance scores for different system components"""
        scores = {}
        
        # Hormone system score
        if hormone_levels:
            hormone_balance = 1.0 - np.std(list(hormone_levels.values()))
            scores['hormone_system'] = max(0.0, min(1.0, hormone_balance))
        
        # Neural system score
        if neural_performance:
            neural_score = neural_performance.get('performance_summary', {}).get('overall_score', 0.5)
            scores['neural_system'] = neural_score
        
        # Genetic system score
        if genetic_triggers:
            success_rate = genetic_triggers.get('success_rate', 0.5)
            scores['genetic_system'] = success_rate
        
        # Resource efficiency score
        if resource_usage:
            cpu_efficiency = 1.0 - (resource_usage.get('cpu_percent', 0.0) / 100.0)
            memory_efficiency = 1.0 - (resource_usage.get('memory_percent', 0.0) / 100.0)
            scores['resource_efficiency'] = (cpu_efficiency + memory_efficiency) / 2.0
        
        # Overall system score
        if scores:
            scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def _detect_anomalies_in_metrics(self, hormone_levels: Dict[str, float],
                                   neural_performance: Dict[str, Any],
                                   genetic_triggers: Dict[str, Any],
                                   resource_usage: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in system metrics"""
        anomalies = []
        
        # Check for hormone spikes
        for hormone, level in hormone_levels.items():
            if level > self.anomaly_thresholds['hormone_spike']:
                anomalies.append({
                    'type': 'hormone_spike',
                    'component': hormone,
                    'value': level,
                    'threshold': self.anomaly_thresholds['hormone_spike'],
                    'severity': 'high' if level > 0.9 else 'medium'
                })
        
        # Check for performance drops
        if neural_performance:
            overall_score = neural_performance.get('performance_summary', {}).get('overall_score', 1.0)
            if overall_score < self.anomaly_thresholds['performance_drop']:
                anomalies.append({
                    'type': 'performance_drop',
                    'component': 'neural_system',
                    'value': overall_score,
                    'threshold': self.anomaly_thresholds['performance_drop'],
                    'severity': 'high'
                })
        
        # Check for high resource usage
        if resource_usage:
            cpu_usage = resource_usage.get('cpu_percent', 0.0) / 100.0
            memory_usage = resource_usage.get('memory_percent', 0.0) / 100.0
            
            if cpu_usage > self.anomaly_thresholds['resource_high']:
                anomalies.append({
                    'type': 'high_cpu_usage',
                    'component': 'system',
                    'value': cpu_usage,
                    'threshold': self.anomaly_thresholds['resource_high'],
                    'severity': 'medium'
                })
            
            if memory_usage > self.anomaly_thresholds['resource_high']:
                anomalies.append({
                    'type': 'high_memory_usage',
                    'component': 'system',
                    'value': memory_usage,
                    'threshold': self.anomaly_thresholds['resource_high'],
                    'severity': 'high'
                })
        
        return anomalies
    
    def _calculate_system_health(self, performance_scores: Dict[str, float],
                               anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall system health score"""
        if not performance_scores:
            return 0.5
        
        # Base health from performance scores
        base_health = performance_scores.get('overall', 0.5)
        
        # Penalize for anomalies
        anomaly_penalty = 0.0
        for anomaly in anomalies:
            severity_multiplier = {
                'low': 0.05,
                'medium': 0.1,
                'high': 0.2,
                'critical': 0.4
            }
            penalty = severity_multiplier.get(anomaly.get('severity', 'low'), 0.05)
            anomaly_penalty += penalty
        
        # Apply penalty
        health = max(0.0, base_health - anomaly_penalty)
        
        return health
    
    def _detect_anomalies(self, metrics: SystemMetrics) -> List[AnomalyAlert]:
        """Detect anomalies and create alerts"""
        alerts = []
        
        for anomaly in metrics.anomalies:
            alert = AnomalyAlert(
                severity=anomaly.get('severity', 'medium'),
                component=anomaly.get('component', 'unknown'),
                message=f"{anomaly['type']} detected in {anomaly['component']}",
                timestamp=metrics.timestamp,
                metrics=anomaly,
                recommendations=self._generate_recommendations(anomaly)
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_recommendations(self, anomaly: Dict[str, Any]) -> List[str]:
        """Generate recommendations for anomaly resolution"""
        recommendations = []
        
        anomaly_type = anomaly.get('type', '')
        
        if anomaly_type == 'hormone_spike':
            recommendations.extend([
                "Check hormone system for feedback loops",
                "Review hormone production triggers",
                "Consider hormone level normalization"
            ])
        elif anomaly_type == 'performance_drop':
            recommendations.extend([
                "Switch to algorithmic implementation",
                "Retrain neural models",
                "Check for resource constraints"
            ])
        elif anomaly_type in ['high_cpu_usage', 'high_memory_usage']:
            recommendations.extend([
                "Reduce computational load",
                "Optimize resource usage",
                "Consider scaling down operations"
            ])
        
        return recommendations
    
    def _handle_anomalies(self, alerts: List[AnomalyAlert]):
        """Handle detected anomalies"""
        for alert in alerts:
            alert_id = f"{alert.component}_{alert.timestamp.isoformat()}"
            
            if alert_id not in self.active_alerts:
                self.active_alerts.add(alert_id)
                self.anomaly_history.append(alert)
                
                # Log the alert
                self.logger.warning(f"Anomaly detected: {alert.message} (Severity: {alert.severity})")
                
                # Trigger external alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")
    
    def _update_performance_history(self, metrics: SystemMetrics):
        """Update performance history tracking"""
        for component, score in metrics.performance_scores.items():
            self.performance_history[component].append(score)
            
            # Keep only recent history
            if len(self.performance_history[component]) > 1000:
                self.performance_history[component] = self.performance_history[component][-1000:]
    
    def _generate_visualization_data(self) -> List[VisualizationData]:
        """Generate visualization data for dashboard"""
        viz_data = []
        
        if not self.current_metrics:
            return viz_data
        
        # Hormone levels chart
        hormone_data = {
            'labels': list(self.current_metrics.hormone_levels.keys()),
            'values': list(self.current_metrics.hormone_levels.values()),
            'colors': self._generate_hormone_colors()
        }
        viz_data.append(VisualizationData(
            chart_type='hormone_levels',
            data=hormone_data,
            options={'title': 'Hormone Levels', 'type': 'bar'},
            timestamp=self.current_metrics.timestamp
        ))
        
        # Performance scores chart
        performance_data = {
            'labels': list(self.current_metrics.performance_scores.keys()),
            'values': list(self.current_metrics.performance_scores.values()),
            'colors': self._generate_performance_colors()
        }
        viz_data.append(VisualizationData(
            chart_type='performance_scores',
            data=performance_data,
            options={'title': 'Performance Scores', 'type': 'radar'},
            timestamp=self.current_metrics.timestamp
        ))
        
        # Resource usage chart
        resource_data = {
            'labels': list(self.current_metrics.resource_usage.keys()),
            'values': list(self.current_metrics.resource_usage.values()),
            'colors': self._generate_resource_colors()
        }
        viz_data.append(VisualizationData(
            chart_type='resource_usage',
            data=resource_data,
            options={'title': 'Resource Usage', 'type': 'doughnut'},
            timestamp=self.current_metrics.timestamp
        ))
        
        # System health trend
        if len(self.metrics_history) > 1:
            health_trend = [m.system_health for m in list(self.metrics_history)[-50:]]
            trend_data = {
                'labels': [f"T{i}" for i in range(len(health_trend))],
                'values': health_trend,
                'color': '#4CAF50'
            }
            viz_data.append(VisualizationData(
                chart_type='system_health_trend',
                data=trend_data,
                options={'title': 'System Health Trend', 'type': 'line'},
                timestamp=self.current_metrics.timestamp
            ))
        
        return viz_data
    
    def _generate_hormone_colors(self) -> List[str]:
        """Generate colors for hormone visualization"""
        return [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA'
        ]
    
    def _generate_performance_colors(self) -> List[str]:
        """Generate colors for performance visualization"""
        return ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
    
    def _generate_resource_colors(self) -> List[str]:
        """Generate colors for resource visualization"""
        return ['#FF5722', '#2196F3', '#4CAF50', '#FFC107']
    
    def _trigger_callbacks(self, metrics: SystemMetrics, alerts: List[AnomalyAlert], viz_data: List[VisualizationData]):
        """Trigger external callbacks"""
        # Trigger visualization callbacks
        for callback in self.visualization_callbacks:
            try:
                callback(viz_data)
            except Exception as e:
                self.logger.error(f"Error in visualization callback: {e}")
    
    def add_alert_callback(self, callback: callable):
        """Add callback for anomaly alerts"""
        self.alert_callbacks.append(callback)
    
    def add_visualization_callback(self, callback: callable):
        """Add callback for visualization data"""
        self.visualization_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics"""
        return self.current_metrics
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_performance_trends(self, component: str, hours: int = 24) -> List[float]:
        """Get performance trends for a specific component"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        trends = []
        for metrics in recent_metrics:
            if component in metrics.performance_scores:
                trends.append(metrics.performance_scores[component])
        
        return trends
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies in the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.anomaly_history if a.timestamp > cutoff_time]
        
        severity_counts = defaultdict(int)
        component_counts = defaultdict(int)
        
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
            component_counts[alert.component] += 1
        
        return {
            'total_anomalies': len(recent_alerts),
            'severity_distribution': dict(severity_counts),
            'component_distribution': dict(component_counts),
            'most_common_anomaly': max(component_counts.items(), key=lambda x: x[1])[0] if component_counts else None
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics data in specified format"""
        if format == 'json':
            data = {
                'current_metrics': self.current_metrics.__dict__ if self.current_metrics else None,
                'metrics_history': [m.__dict__ for m in list(self.metrics_history)[-100:]],
                'anomaly_summary': self.get_anomaly_summary(),
                'performance_trends': {
                    component: self.get_performance_trends(component)
                    for component in ['overall', 'hormone_system', 'neural_system', 'genetic_system']
                }
            }
            return json.dumps(data, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'monitoring_active': self.monitoring_active,
            'current_health': self.current_metrics.system_health if self.current_metrics else 0.0,
            'total_metrics_collected': len(self.metrics_history),
            'active_alerts': len(self.active_alerts),
            'total_anomalies': len(self.anomaly_history),
            'performance_summary': self.current_metrics.performance_scores if self.current_metrics else {},
            'anomaly_summary': self.get_anomaly_summary(hours=1)
        } 