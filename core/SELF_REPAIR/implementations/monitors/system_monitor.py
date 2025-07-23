#!/usr/bin/env python3
"""
System Monitor for Self-Repair
@{CORE.SELF_REPAIR.MONITOR.001} System monitoring component for self-repair.
#{monitoring,metrics,anomaly,detection}
Δ(β(monitoring_implementation))
"""

import asyncio
import logging
import time
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Import core components
from core.src.mcp.core_system import MCPCoreSystem
from core.src.mcp.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class MetricSnapshot:
    """Immutable snapshot of system metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    active_components: int
    component_metrics: Dict[str, Dict[str, float]]
    custom_metrics: Dict[str, float]

class SystemMonitor:
    """
    System monitoring component for self-repair.
    
    Collects and analyzes system metrics to detect anomalies.
    Uses statistical models for baseline comparison.
    """
    
    def __init__(self, core_system: MCPCoreSystem):
        """Initialize the system monitor."""
        self.core_system = core_system
        self.performance_monitor = core_system.performance_monitor
        
        # Metric history
        self.metric_history: List[MetricSnapshot] = []
        self.max_history_size = 1000
        
        # Statistical baselines
        self.metric_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, float] = {
            "cpu_usage": 0.9,
            "memory_usage": 0.85,
            "disk_usage": 0.9,
            "network_usage": 0.8
        }
        
        logger.info("System Monitor initialized")
    
    async def start_monitoring(self):
        """Start the monitoring process."""
        logger.info("Starting system monitoring")
        
        while not self.core_system._shutdown_event.is_set():
            try:
                # Collect current metrics
                snapshot = self._collect_metrics()
                
                # Add to history
                self._add_to_history(snapshot)
                
                # Update baselines periodically
                if len(self.metric_history) % 10 == 0:
                    self._update_baselines()
                
                # Wait before next collection
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(10)
    
    def _collect_metrics(self) -> MetricSnapshot:
        """Collect current system metrics."""
        # Get core metrics
        cpu_usage = self.performance_monitor.get_cpu_usage()
        memory_usage = self.performance_monitor.get_memory_usage()
        disk_usage = self.performance_monitor.get_disk_usage()
        network_usage = self.performance_monitor.get_network_usage()
        
        # Get component metrics
        component_metrics = {}
        active_components = 0
        
        for component_id, component in self.core_system.components.items():
            if component.is_active:
                active_components += 1
                component_metrics[component_id] = {
                    "cpu": component.get_cpu_usage(),
                    "memory": component.get_memory_usage(),
                    "error_rate": component.get_error_rate(),
                    "response_time": component.get_response_time()
                }
        
        # Get custom metrics
        custom_metrics = self.performance_monitor.get_custom_metrics()
        
        # Create snapshot
        return MetricSnapshot(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_usage=network_usage,
            active_components=active_components,
            component_metrics=component_metrics,
            custom_metrics=custom_metrics
        )
    
    def _add_to_history(self, snapshot: MetricSnapshot):
        """Add metric snapshot to history with size limit."""
        self.metric_history.append(snapshot)
        
        # Limit history size
        if len(self.metric_history) > self.max_history_size:
            self.metric_history = self.metric_history[-self.max_history_size:]
    
    def _update_baselines(self):
        """Update statistical baselines from history."""
        if len(self.metric_history) < 10:
            return
        
        # Calculate baselines for core metrics
        cpu_values = [s.cpu_usage for s in self.metric_history]
        memory_values = [s.memory_usage for s in self.metric_history]
        disk_values = [s.disk_usage for s in self.metric_history]
        network_values = [s.network_usage for s in self.metric_history]
        
        self.metric_baselines["cpu_usage"] = {
            "mean": np.mean(cpu_values),
            "std": np.std(cpu_values),
            "min": np.min(cpu_values),
            "max": np.max(cpu_values)
        }
        
        self.metric_baselines["memory_usage"] = {
            "mean": np.mean(memory_values),
            "std": np.std(memory_values),
            "min": np.min(memory_values),
            "max": np.max(memory_values)
        }
        
        self.metric_baselines["disk_usage"] = {
            "mean": np.mean(disk_values),
            "std": np.std(disk_values),
            "min": np.min(disk_values),
            "max": np.max(disk_values)
        }
        
        self.metric_baselines["network_usage"] = {
            "mean": np.mean(network_values),
            "std": np.std(network_values),
            "min": np.min(network_values),
            "max": np.max(network_values)
        }
        
        # Calculate baselines for component metrics
        # Implementation would continue...