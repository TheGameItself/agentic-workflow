"""
ResourceOptimizationEngine: Dynamic resource optimization based on workload patterns.

This module implements a biologically-inspired resource optimization system that
dynamically adjusts hormone production rates, triggers memory consolidation,
schedules background training, and adapts to resource constraints based on
workload patterns and system state.
"""

import logging
import time
import asyncio
import psutil
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
from collections import deque

from src.mcp.hormone_system_controller import HormoneSystemController
from src.mcp.brain_state_aggregator import BrainStateAggregator
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
from src.mcp.workload_pattern_analyzer import WorkloadPatternAnalyzer, WorkloadPattern, ResourcePrediction


@dataclass
class ResourceMetrics:
    """Resource usage metrics for the system."""
    cpu_usage: float = 0.0  # CPU usage percentage (0-100)
    memory_usage: float = 0.0  # Memory usage percentage (0-100)
    memory_available: int = 0  # Available memory in bytes
    disk_usage: float = 0.0  # Disk usage percentage (0-100)
    network_usage: float = 0.0  # Network usage in bytes/sec
    io_wait: float = 0.0  # IO wait percentage (0-100)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResourceConstraints:
    """Resource constraints for the system."""
    max_cpu_usage: float = 80.0  # Maximum CPU usage percentage
    max_memory_usage: float = 80.0  # Maximum memory usage percentage
    max_disk_usage: float = 90.0  # Maximum disk usage percentage
    max_network_usage: float = 1000000  # Maximum network usage in bytes/sec
    priority_lobes: List[str] = field(default_factory=list)  # Lobes to prioritize


@dataclass
class TrainingSchedule:
    """Schedule for background training tasks."""
    tasks: List[Dict[str, Any]] = field(default_factory=list)  # Training tasks
    priority: Dict[str, int] = field(default_factory=dict)  # Task priorities
    resource_allocation: Dict[str, float] = field(default_factory=dict)  # Resource allocation
    start_times: Dict[str, str] = field(default_factory=dict)  # Task start times
    duration_estimates: Dict[str, float] = field(default_factory=dict)  # Duration estimates


@dataclass
class AdaptationPlan:
    """Plan for adapting to resource constraints."""
    hormone_adjustments: Dict[str, float] = field(default_factory=dict)  # Hormone level adjustments
    lobe_priority_changes: Dict[str, int] = field(default_factory=dict)  # Lobe priority changes
    memory_consolidation_targets: List[str] = field(default_factory=list)  # Memory consolidation targets
    background_task_adjustments: Dict[str, str] = field(default_factory=dict)  # Task adjustments
    estimated_resource_savings: Dict[str, float] = field(default_factory=dict)  # Estimated resource savings


@dataclass
class RecoveryState:
    """State information for recovery from resource constraints."""
    previous_hormone_levels: Dict[str, float] = field(default_factory=dict)  # Previous hormone levels
    optimal_hormone_levels: Dict[str, float] = field(default_factory=dict)  # Optimal hormone levels
    recovery_start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    recovery_duration: float = 300.0  # Recovery duration in seconds
    recovery_progress: float = 0.0  # Recovery progress (0-1)


class ResourceOptimizationEngine:
    """
    Engine for optimizing resource usage based on workload patterns and system constraints.
    
    This engine dynamically adjusts hormone production rates, triggers memory consolidation,
    schedules background training, and adapts to resource constraints based on workload
    patterns and system state.
 """
 
    def __init__(self, hormone_controller=None, brain_state_aggregator=None, event_bus=None):
        """
        Initialize the resource optimization engine.
        Args:
            hormone_controller: Hormone controller instance
            brain_state_aggregator: Brain state aggregator instance
            event_bus: Event bus for system events
        """
        self.logger = logging.getLogger("ResourceOptimizationEngine")
        self.hormone_controller = hormone_controller
        self.brain_state_aggregator = brain_state_aggregator
        self.event_bus = event_bus
        self.metrics_history = deque(maxlen=1000)
        self.current_constraints = ResourceConstraints()
        self.hormone_adjustment_active = False
        self.hormone_baseline_levels = {}
        self.hormone_adjustment_factors = {}
        self.memory_consolidation_active = False
        self.last_consolidation_time = None
        self.consolidation_cooldown = 300  # 5 minutes
        self.in_recovery_mode = False
        self.recovery_state = None
        self.training_schedule = None
        self.current_metrics = self._collect_resource_metrics()
        self.pattern_analyzer = WorkloadPatternAnalyzer()
        self.current_prediction = None
        self.last_prediction_time = datetime.now() - timedelta(minutes=5)
        self.adaptation_plans = {}
        self.active_adaptations = set()
        self.memory_consolidation_threshold = 0.8  # Default threshold for memory consolidation
        self.active_training_tasks = set()
        self._setup_event_handlers()
        self.logger.info("ResourceOptimizationEngine initialized.")
    
    def _setup_event_handlers(self):
        """Set up event handlers"""
        if self.event_bus:
            self.event_bus.subscribe("memory_pressure_detected", self._handle_memory_pressure)
            self.event_bus.subscribe("resource_recovery_detected", self._handle_resource_recovery)
            
            self.logger.info("Resource event handlers registered")
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """
        Collect current resource usage metrics.
        
        Returns:
            ResourceMetrics object
        """
        try:
            # Collect CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Collect memory usage
            memory_usage = psutil.virtual_memory().percent
            memory_available = psutil.virtual_memory().available
            
            # Collect disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Collect network usage (simplified)
            network_usage = 0.0
            try:
                net_io = psutil.net_io_counters()
                network_usage = net_io.bytes_sent + net_io.bytes_recv
            except:
                pass  # Network metrics might not be available
            
            # Collect IO wait
            io_wait = 0.0
            try:
                io_wait = psutil.cpu_times().idle / psutil.cpu_times().total * 100
            except:
                pass  # RMS usage might not be available
            
            metrics = ResourceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_available=memory_available,
                disk_usage=disk_usage,
                network_usage=network_usage,
                io_wait=io_wait
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Update pattern analyzer
            self.pattern_analyzer.update(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_usage=network_usage
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting resource metrics: {e}")
           
            return ResourceMetrics()
    
    def update(self):
        """
        Update the resource optimization engine.
        """
        # Collect current metrics
        self.current_metrics = self._collect_resource_metrics()
        
        if self._is_high_computational_load():
            self._handle_high_computational_load()
        
        if self._is_memory_pressure():
            self._handle_memory_pressure_internal()
        
        if self._is_system_idle():
            self._handle_system_idle_internal()
        
        if self.in_recovery_mode:
            self._update_recovery()
        
        # Update workload patterns and predictions
        self._update_workload_patterns()
        
        # Update prediction if enough time has passed
        if (datetime.now() - self.last_prediction_time).total_seconds() > 300:
            self.last_prediction_time = datetime.now()
        
        # Log updates periodically
        if len(self.metrics_history) % 50 == 0:
            self._log_current_state()
    
    def _is_high_computational_load(self) -> bool:
        """
        Check if the system is under high computational load.
        
        Returns:
            True if high computational load detected, otherwise False
        """
        # Check if CPU usage is above threshold
        cpu_threshold = self.current_constraints.max_cpu_usage
        return self.current_metrics.cpu_usage > cpu_threshold
    
    def _is_memory_pressure(self) -> bool:
        """
        Check if the system is under memory pressure.
        
        Returns:
            True if memory pressure detected, otherwise False
        """
        # Check if memory usage is above threshold
        return self.current_metrics.memory_usage > 80.0
    
    def _is_system_idle(self) -> bool:
        """
        Check if the system is idle.
        
        Returns:
            True if system is idle, False otherwise
        """
        # Check if CPU usage is below idle threshold
        return self.current_metrics.cpu_usage < 10.0
    
    def _handle_high_computational_load(self):
        """Handle high computational load."""
        if not self.hormone_adjustment_active:
            # Store baseline hormone levels if not already stored
            if self.hormone_controller:
                self.hormone_baseline_levels = self.hormone_controller.get_hormone_levels()
            
            # Calculate adjustment factors based on load
            load_factor = min(1.0, self.current_metrics.cpu_usage / 100.0)
            
            # Higher load = more reduction in non-essential hormones
            reduction_factor = 0.5 + (0.4 * load_factor)  # 0.5 to 0.9 reduction
            
            # Apply adjustments
            self.hormone_adjustment_factors = {
                # Reduce these hormones during high load
                "dopamine": 1.0 - (0.3 * load_factor),  # Reduce reward signal
                "serotonin": 1.0 - (0.2 * load_factor),  # Reduce confidence signal
                "growth_hormone": 1.0 - (0.3 * load_factor),  # Reduce growth hormone
                "cortisol": 1.0 + (0.3 * load_factor),  # Increase cortisol
                "adrenaline": 1.0 + (0.4 * load_factor)  # Increase adrenaline
            }
            
            self._apply_hormone_adjustments()
            
            # Set adjustment active 
            self.hormone_adjustment_active = True
            
            self.logger.info(f"Activated hormone adjustment (CPU: {self.current_metrics.cpu_usage:.1f}%)")
            
            # Emit event
            if self.event_bus:
                self.event_bus.emit(
                    "resource_optimization_activated",
                    {
                        "type": "high_load",
                        "cpu_usage": self.current_metrics.cpu_usage,
                        "memory_usage": self.current_metrics.memory_usage,
                        "disk_usage": self.current_metrics.disk_usage,
                        "network_usage": self.current_metrics.network_usage
                    }
                )
    
    def _handle_memory_pressure_internal(self):
        """Internal handler for memory pressure."""
        current_time = datetime.now()
        if self.last_consolidation_time is None:
            time_since_last = float('inf')
        else:
            time_since_last = (current_time - self.last_consolidation_time).total_seconds()
        if time_since_last > self.consolidation_cooldown:
            self.trigger_memory_consolidation(
                self.current_metrics.memory_usage / 100.0,
                self.memory_consolidation_threshold
            )
            self.last_consolidation_time = current_time
    
    def _handle_system_idle_internal(self):
        """Internal handler for system idle state."""
        # Schedule background training
        if self.training_schedule:
            available_resources = ResourceMetrics(
                cpu_usage=self.current_metrics.cpu_usage,
                memory_usage=self.current_metrics.memory_usage,
                memory_available=self.current_metrics.memory_available,
                disk_usage=self.current_metrics.disk_usage,
                network_usage=self.current_metrics.network_usage,
                io_wait=self.current_metrics.io_wait
            )
            
            self.schedule_background_training(available_resources)
    
    def _handle_resource_constraint(self, data: Dict[str, Any]):
        """Handle resource constraint event.
        
        Args:
            data: Event data containing constraint information
        """
        # Extract constraint information
        constraint_type = data.get("constraint_type", "unknown")
        max_cpu_usage = data.get("max_cpu_usage", self.current_constraints.max_cpu_usage)
        max_memory_usage = data.get("max_memory_usage", self.current_constraints.max_memory_usage)
        max_disk_usage = data.get("max_disk_usage", self.current_constraints.max_disk_usage)
        max_network_usage = data.get("max_network_usage", self.current_constraints.max_network_usage)
        priority_lobes = data.get("priority_lobes", self.current_constraints.priority_lobes)
        
        # Create constraint object
        constraints = ResourceConstraints(
            max_cpu_usage=max_cpu_usage,
            max_memory_usage=max_memory_usage,
            max_disk_usage=max_disk_usage,
            max_network_usage=max_network_usage,
            priority_lobes=priority_lobes
        )
        
        # Update current constraints
        self.current_constraints = constraints
        
        # Adapt to new constraints
        adaptation_plan = self.adapt_to_resource_constraints(constraints)
        
        self.logger.info(f"Adapting to resource constraints: {constraint_type}")
        
        # Log adaptation plan
        self.logger.info(f"Adaptation plan: {len(adaptation_plan.hormone_adjustments)}, {len(adaptation_plan.lobe_priority_changes)}, {len(adaptation_plan.memory_consolidation_targets)}, {len(adaptation_plan.background_task_adjustments)}")
    
    def _handle_memory_pressure(self, data: Dict[str, Any]):
        """Handle memory pressure event.
        
        Args:
            data: Event data containing memory pressure information
        """
        threshold = data.get("threshold", self.memory_consolidation_threshold)
        
        # Trigger memory consolidation
        self.trigger_memory_consolidation(data.get("memory_usage", 0.0), threshold)
    
    def _handle_system_idle(self, data: Dict[str, Any]):
        """Handle system idle event.
        
        Args:
            data: Event data containing system idle information
        """
        # Extract available resources
        available_resources = ResourceMetrics(
            cpu_usage=data.get("cpu_usage", 0.0),
            memory_usage=data.get("memory_usage", 0.0),
            memory_available=data.get("memory_available", 0),
            disk_usage=data.get("disk_usage", 0.0),
            network_usage=data.get("network_usage", 0.0),
            io_wait=data.get("io_wait", 0.0)
        )
        
        # Schedule background training
        self.schedule_background_training(available_resources)
    
    def _handle_high_activity(self, data: Dict[str, Any]):
        """Handle high activity event.
        
        Args:
            data: Event data containing active lobes
        """
        # Extract active lobes
        active_lobes = data.get("active_lobes", {})
        
        # Prioritize resources for active lobes
        if active_lobes:
            resource_allocation = self.prioritize_lobe_resources(list(active_lobes.keys()), active_lobes)
            
            self.logger.info(f"Prioritized resource allocation: {resource_allocation}")
            
            # Emit updated allocation event
            if self.event_bus:
                self.event_bus.emit(
                    "resource_allocation_updated",
                    {
                        "allocation": resource_allocation,
                        "active_lobes": active_lobes
                    }
                )
    
    def _handle_resource_recovery(self, data: Dict[str, Any]):
        """Handle resource recovery event.
        
        Args:
            data: Event data containing recovery information
        """
        previous_hormone_levels = data.get("previous_hormone_levels", {})
        optimal_hormone_levels = data.get("optimal_hormone_levels", {})
        
        # Create recovery state
        recovery_state = RecoveryState(
            previous_hormone_levels=previous_hormone_levels,
            optimal_hormone_levels=optimal_hormone_levels
        )
        
        self.restore_optimal_levels(recovery_state)
    
    def _apply_hormone_adjustments(self):
        """Apply hormone adjustments based on current adjustment factors."""
        if not self.hormone_controller or not self.hormone_baseline_levels:
            return
        
        current_levels = self.hormone_controller.get_hormone_levels() if self.hormone_controller else {}
        for hormone, factor in self.hormone_adjustment_factors.items():
            if hormone in current_levels:
                baseline = self.hormone_baseline_levels.get(hormone, 0.5)
                target_level = baseline
                diff = target_level - current_levels[hormone]
                adjustment = diff * 0.2
                new_level = current_levels[hormone] + adjustment
                new_level = max(0.0, min(1.0, new_level))
                if self.hormone_controller:
                    self.hormone_controller.set_hormone_level(hormone, new_level)
                self.logger.debug(f"Adjusted {hormone} from {current_levels[hormone]:.2f} to {new_level:.2f}")
    
    def _update_recovery(self):
        """Update recovery process if active."""
        if not self.recovery_state:
            return
        
        start_time = datetime.fromisoformat(self.recovery_state.recovery_start_time)
        elapsed = (datetime.now() - start_time).total_seconds()
        total_duration = self.recovery_state.recovery_duration
        
        # Calculate progress
        progress = min(1.0, elapsed / total_duration)
        self.recovery_state.recovery_progress = progress
        
        # Get current hormone levels
        current_levels = self.hormone_controller.get_hormone_levels() if self.hormone_controller else {}
        
        # Apply recovery
        for hormone, optimal in self.recovery_state.optimal_hormone_levels.items():
            if hormone in current_levels:
                # Calculate target level
                target_level = optimal
                
                # Calculate difference from current
                diff = target_level - current_levels[hormone]
                
                # Apply adjustment
                adjustment = diff * 0.2
                new_level = current_levels[hormone] + adjustment
                
                # Clamp to valid range
                new_level = max(0.0, min(1.0, new_level))
                
                # Apply to controller
                if self.hormone_controller:
                    self.hormone_controller.set_hormone_level(hormone, new_level)
                
                self.logger.debug(f"Recovered {hormone} from {current_levels[hormone]:.2f} to {new_level:.2f}")
        
        # Check if recovery is complete
        if progress >= 1.0:
            self.in_recovery_mode = False
            self.recovery_state = None
            
            # Emit event
            if self.event_bus:
                self.event_bus.emit(
                    "resource_recovery_completed",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "duration": elapsed
                    }
                )
    
    def _update_workload_patterns(self):
        """Update workload patterns and predictions."""
        # Get current pattern from analyzer
        self.current_prediction = self.pattern_analyzer.get_resource_prediction()
        if self.current_prediction:
            self.logger.info(f"Generated resource prediction: {self.current_prediction}")
            # Emit prediction event
            if self.event_bus:
                self.event_bus.emit(
                    "resource_prediction_generated",
                    {
                        "prediction": {
                            "cpu_trend": "stable" if all(v <= 0.5 for v in self.current_prediction.predicted_cpu.values()) else "increasing",
                            "memory_trend": "stable" if all(v <= 0.5 for v in self.current_prediction.predicted_memory.values()) else "increasing",
                            "confidence": self.current_prediction.confidence,
                            "recommended_actions": self.current_prediction.recommended_actions,
                        },
                        "prediction_horizon": self.current_prediction.prediction_horizon,
                    }
                )
            # Take proactive actions
            self._apply_predictive_optimizations(self.current_prediction)
    
    def _apply_predictive_optimizations(self, prediction: ResourcePrediction):
        """Apply optimizations based on prediction.
        
        Args:
            prediction: ResourcePrediction object
        """
        if prediction.confidence < 0.5:
            return
        
        # Check for predicted high CPU usage
        max_predicted_cpu = max(prediction.predicted_cpu.values()) if prediction.confidence > 0.5 else 0
        
        if max_predicted_cpu > 0:
            self.logger.info(f"Preemptively reducing hormone production due to predicted CPU usage ({max_predicted_cpu})")
            self._handle_high_computational_load()
        
        # Check for predicted high memory usage
        max_predicted_memory = max(prediction.predicted_memory.values()) if prediction.confidence > 0.5 else 0
        
        if max_predicted_memory > 0:
            self.logger.info(f"Preemptively reducing hormone production due to predicted memory usage ({max_predicted_memory})")
            self._handle_high_computational_load()
    
    def _autocorrelation(self, series: List[float]) -> List[float]:
        """Calculate autocorrelation of series.
        
        Args:
            series: Time series data
            
        Returns:
            List of autocorrelation values
        """
        # Center data
        data = series - np.mean(series)
        
        # Calculate autocorrelation
        result = np.correlate(data, data, mode='full')
        
        # Normalize and take only the positive lags
        result = result[len(data) - 1:] / np.max(result)
        
        return result.tolist()
    
    def _find_peaks(self, series: List[float], threshold: float) -> List[int]:
        """Find peaks in a series.
        
        Args:
            series: Series data
            threshold: Threshold for peak detection
            
        Returns:
            List of peak indices
        """
        peaks = []
        
        # Skip first few elements
        for i in range(5, len(series) - 1):
            if series[i] > threshold and series[i] > series[i-1] and series[i] > series[i+1]:
                peaks.append(i)
                
        return peaks
    
    def _check_burstiness(self, series: List[float], threshold: float) -> bool:
        """Check if series is bursty.
        
        Args:
            series: Series data
            threshold: Threshold for burstiness detection
            
        Returns:
            True if series is bursty, otherwise False
        """
        if not series:
            return False
        
        # Calculate mean and standard deviation
        mean = np.mean(series)
        std = np.std(series)
        
        # Calculate coefficient of variation
        cv = std / mean if mean > 0 else 0
        
        # Check for burstiness
        return cv > threshold
    
    def get_active_lobes(self) -> Dict[str, float]:
        """Get currently active lobes and their activity levels.
        
        Returns:
            Dictionary mapping lobe names to activity levels
        """
        if self.brain_state_aggregator:
            # This is a simplified approach 
            brain_state = self.brain_state_aggregator.get_environment_state()
        
            # Extract active lobes (simplified)
            active_lobes = {}
            for lobe_name, lobe_state in brain_state.get("lobes", {}):
                if isinstance(lobe_state, dict) and "activity" in lobe_state:
                    active_lobes[lobe_name] = lobe_state["activity"]
            
            return active_lobes
        
        # Return empty dict if no brain state aggregator
        return {}
    
    def _log_current_state(self):
        """Log current resource optimization state."""
        self.logger.info(f"CPU: {self.current_metrics.cpu_usage:.1f}%, Memory: {self.current_metrics.memory_usage:.1f}%, Disk: {self.current_metrics.disk_usage:.1f}%, Network: {self.current_metrics.network_usage:.1f}%, IO Wait: {self.current_metrics.io_wait:.1f}%")
        
        if self.hormone_adjustment_active:
            self.logger.info("Hormone adjustment active")
        
        if self.memory_consolidation_active:
            self.logger.info("Memory consolidation active")
        
        if self.in_recovery_mode:
            self.logger.info("Recovery active")
    
    def trigger_memory_consolidation(self, memory_usage: float, threshold: float):
        """Trigger memory consolidation.
        
        Args:
            memory_usage: Current memory usage (0-1)
            threshold: Memory usage threshold (0-1)
        """
        if self.memory_consolidation_active:
            return
        
        # Check severity
        severity = min(1.0, (memory_usage - threshold) / (1.0 - threshold) * 0.2)
        
        # Adjust hormones
        if self.hormone_controller:
            # Increase vasopressin to enhance memory consolidation
            self.hormone_controller.set_hormone_level("vasopressin", 0.8 * severity)
            
            # Reduce growth hormone to slow down memory consolidation
            current_levels = self.hormone_controller.get_hormone_levels()
            new_level = max(0.1, current_levels["growth_hormone"] - 0.3 * severity)
            self.hormone_controller.set_hormone_level("growth_hormone", new_level)
        
            self.logger.info(f"Memory consolidation triggered (severity: {severity:.2f})")
            
            # Emit event
            if self.event_bus:
                self.event_bus.emit(
                    "memory_consolidation_triggered",
                    {
                        "threshold": threshold,
                        "severity": severity,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        # Reset flag after a delay
        def reset_flag():
            self.memory_consolidation_active = False
        
        # Schedule reset after 30 seconds
        timer = threading.Timer(30.0, reset_flag)
        timer.daemon = True
        timer.start()
    
    def schedule_background_training(self, available_resources: ResourceMetrics):
        """Schedule background training tasks based on available resources.
        
        Args:
            available_resources: Available system resources
            
        Returns:
            TrainingSchedule object
        """
        # Calculate available capacity
        available_cpu = max(0.0, 70.0 - available_resources.cpu_usage)
        available_memory = max(0.0, 70.0 - available_resources.memory_usage)
        
        # Skip if not enough resources available
        if available_cpu < 30.0 or available_memory < 30.0:
            return None
        
        # Create schedule
        schedule = TrainingSchedule()
        
        # Add tasks
        if available_cpu > 30.0 and available_memory > 30.0:
            schedule.tasks.append({
                "task_id": "neural_model_training",
                "resource_requirements": {
                    "cpu": 25.0,
                    "memory": 20.0
                },
                "estimated_duration": 300.0,  # 5 minutes
                "priority": 1
            })
            
            schedule.resource_allocation["neural_model_training"] = 25.0
            schedule.start_times["neural_model_training"] = datetime.now().isoformat()
            schedule.duration_estimates["neural_model_training"] = 300.0
            
            self.active_training_tasks.add("neural_model_training")
        elif available_cpu > 5.0 and available_memory > 5.0:
            # Medium re
            schedule.tasks.append({
                "task_id": "incremental_training",
                "resource_requirements": {
                    "cpu": 5.0,
                    "memory": 5.0
                },
                "priority": 2
            })
            
            schedule.priority["incremental_training"] = 2
            schedule.resource_allocation["incremental_training"] = 5.0
            schedule.start_times["incremental_training"] = datetime.now().isoformat()
            schedule.duration_estimates["incremental_training"] = 120.0
            
            self.active_training_tasks.add("incremental_training")
        
        # Add pattern analysis task (light)
        schedule.tasks.append({
            "task_id": "pattern_analysis",
            "resource_requirements": {
                "cpu": 5.0,
                "memory": 5.0
            },
            "priority": 3
        })
        
        schedule.priority["pattern_analysis"] = 3
        schedule.resource_allocation["pattern_analysis"] = 5.0
        schedule.start_times["pattern_analysis"] = datetime.now().isoformat()
        schedule.duration_estimates["pattern_analysis"] = 60.0
        
        # Store schedule
        self.training_schedule = schedule
        
        self.logger.info(f"Scheduled background training (available CPU: {available_cpu:.1f}%, available Memory: {available_memory:.1f}%)")
        
        # Emit event
        if self.event_bus:
            self.event_bus.emit(
                "background_training_scheduled",
                {
                    "le": {
                        "tasks": [task["task_id"] for task in schedule.tasks],
                        "priority": schedule.priority
                    },
                    "available_resources": {
                        "cpu": available_cpu,
                        "memory": available_memory
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return schedule
    
    def predict_workload_pattern(self) -> WorkloadPattern:
        """Predict workload pattern based on data.
        
        Returns:
            WorkloadPattern object with predicted pattern
        """
        pattern = self.pattern_analyzer.get_current_pattern()
        if pattern is not None:
            return pattern
        else:
            return WorkloadPattern(pattern_type="steady", confidence=0.0)
    
    def adapt_to_resource_constraints(self, constraints: ResourceConstraints) -> AdaptationPlan:
        """Adapt system behavior to resource constraints.
        
        Args:
            constraints: ResourceConstraints object
            
        Returns:
            AdaptationPlan object
        """
        # Create adaptation plan
        plan = AdaptationPlan()
        
        # Calculate severity
        cpu_severity = max(0.0, min(1.0, (self.current_metrics.cpu_usage - constraints.max_cpu_usage) / (100.0 - constraints.max_cpu_usage)))
        memory_severity = max(0.0, min(1.0, (self.current_metrics.memory_usage - constraints.max_memory_usage) / (100.0 - constraints.max_memory_usage)))
        disk_severity = max(0.0, min(1.0, (self.current_metrics.disk_usage - constraints.max_disk_usage) / (100.0 - constraints.max_disk_usage)))
        
        # Overall severity
        severity = max(cpu_severity, memory_severity, disk_severity)
        
        # Skip if no constraints are violated
        if severity == 0.0:
            return plan
        
        # Adjust hormones based on constraints
        if self.hormone_controller:
            # Get current hormone levels
            current_levels = self.hormone_controller.get_hormone_levels()
            
            # Apply adjustments
            if cpu_severity > 0.0:
                plan.hormone_adjustments["dopamine"] = 1.0 - (0.3 * cpu_severity)
                plan.hormone_adjustments["serotonin"] = 1.0 - (0.2 * cpu_severity)
                plan.hormone_adjustments["growth_hormone"] = 1.0 - (0.3 * cpu_severity)
                plan.hormone_adjustments["cortisol"] = 1.0 + (0.3 * cpu_severity)
                plan.hormone_adjustments["adrenaline"] = 1.0 + (0.4 * cpu_severity)
                
                # Estimate resource savings
                plan.estimated_resource_savings["cpu"] = 10.0 + 20.0 * cpu_severity
            
            if memory_severity > 0.0:
                plan.hormone_adjustments["growth_hormone"] = max(0.1, current_levels["growth_hormone"] - 0.3 * memory_severity)
                
                # Estimate resource savings
                plan.estimated_resource_savings["memory"] = 10.0 + 20.0 * memory_severity
            
            if disk_severity > 0.0:
                plan.hormone_adjustments["cortisol"] = max(0.1, current_levels["cortisol"] - 0.3 * disk_severity)
                
                # Estimate resource savings
                plan.estimated_resource_savings["disk"] = 10.0 + 20.0 * disk_severity
        
        # Adjust lobe priorities
        for lobe in constraints.priority_lobes:
            plan.lobe_priority_changes[lobe] = 10  # Increase priority by 10
        
        # Adjust background tasks
        if self.training_schedule is not None:
            for task in self.training_schedule.tasks:
                task_id = task.get("task_id", "unknown")
                if severity > 0.5:
                    plan.background_task_adjustments[task_id] = "prioritize"
                elif severity > 0.2:
                    plan.background_task_adjustments[task_id] = "use"
                else:
                    plan.background_task_adjustments[task_id] = "reduce"
        
        # Apply adaptation plan
        self._apply_adaptation_plan(plan)
        
        return plan
    
    def _apply_adaptation_plan(self, plan: AdaptationPlan):
        """Apply adaptation plan to the system.
        
        Args:
            plan: AdaptationPlan object
        """
        # Apply hormone adjustments
        if self.hormone_controller:
            for hormone, adjustment in plan.hormone_adjustments.items():
                self.hormone_controller.set_hormone_level(hormone, adjustment)
                self.logger.debug(f"Adjusted {hormone} from {self.hormone_baseline_levels.get(hormone, 0.5):.2f} to {adjustment:.2f}")
        
        # Apply lobe priority changes
        if self.brain_state_aggregator:
            # In a real system, this would involve more complex logic
            pass
        
        # Apply background task adjustments
        for task_id, action in plan.background_task_adjustments.items():
            if action == "prioritize":
                self.logger.info(f"Prioritized background task {task_id}")
            elif action == "use":
                self.logger.info(f"Used background task {task_id}")
            elif action == "reduce":
                self.logger.info(f"Reduced resources for background task {task_id}")
            # Apply action to task
            if task_id in self.active_training_tasks:
                self.active_training_tasks.remove(task_id)
                self.logger.info(f"Removed background task {task_id} from active tasks")
            elif action == "pause":
                self.logger.info(f"Paused background task {task_id} due to resource constraint")
            # Apply action to resource allocation
            if self.training_schedule is not None and task_id in self.training_schedule.resource_allocation:
                self.training_schedule.resource_allocation[task_id] = 0.0
                self.logger.info(f"Reduced resources for background task {task_id}")
        
        # Apply memory consolidation targets
        if plan.memory_consolidation_targets:
            self.memory_consolidation_active = True
            self.logger.info(f"Memory consolidation targets: {plan.memory_consolidation_targets}")
            self.trigger_memory_consolidation(0.0, 0.0)
    
    def restore_optimal_levels(self, recovery_state: RecoveryState):
        """Start recovery process to restore levels.
        
        Args:
            recovery_state: Recovery state information
        """
        self.in_recovery_mode = True
        self.recovery_state = recovery_state
        
        self.logger.info(f"Starting hormone level recovery (duration: {recovery_state.recovery_duration} seconds)")
        
        # Emit event
        if self.event_bus:
            self.event_bus.emit(
                "resource_recovery_started",
                {
                    "recovery_state": {
                        "previous_hormone_levels": recovery_state.previous_hormone_levels,
                        "optimal_hormone_levels": recovery_state.optimal_hormone_levels,
                        "recovery_progress": recovery_state.recovery_progress,
                        "hormone_count": len(recovery_state.previous_hormone_levels)
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def prioritize_lobe_resources(self, active_lobes: List[str], resource_allocation: Dict[str, float]) -> Dict[str, float]:
        """Prioritize lobe resources.
        
        Args:
            active_lobes: List of active lobe names
            resource_allocation: Current resource allocation
            
        Returns:
            Updated resource allocation
        """
        # Start with current allocation or empty dict
        updated_allocation = resource_allocation.copy() if resource_allocation else {}
        
        # Calculate total available resources
        total_resources = 100.0
        allocated_resources = sum(updated_allocation.values())
        available_resources = max(0.0, total_resources - allocated_resources)
        
        # Calculate resources per active lobe
        if active_lobes:
            resources_per_lobe = available_resources / len(active_lobes)
            
            for lobe in active_lobes:
                if lobe in updated_allocation:
                    # Increase existing allocation
                    updated_allocation[lobe] += resources_per_lobe
                else:
                    # New allocation
                    updated_allocation[lobe] = resources_per_lobe
        
        # Normalize allocations to ensure total <= 100%
        total_updated = sum(updated_allocation.values())
        if total_updated > total_resources:
            scale_factor = total_resources / total_updated
            for lobe in updated_allocation:
                updated_allocation[lobe] *= scale_factor
        
        return updated_allocation
    
    def get_current_prediction(self) -> Optional[ResourcePrediction]:
        """Get current resource prediction.
        
        Returns:
            ResourcePrediction object or None if no prediction available
        """
        return self.current_prediction if self.current_prediction is not None else None
    
    def get_adaptation_plan(self, constraint_type: str) -> Optional[AdaptationPlan]:
        """Get the adaptation plan for a specific constraint type.
        
        Args:
            constraint_type: Constraint type
            
        Returns:
            AdaptationPlan object or None if not available
        """
        return self.adaptation_plans.get(constraint_type) if self.adaptation_plans.get(constraint_type) is not None else None


# For threading support
import threading