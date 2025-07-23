"""
PredictiveResourceAllocationIntegration: Integration of predictive resource allocation with resource optimization.

This module integrates the PredictiveResourceAllocation system with the ResourceOptimizationEngine
to provide workload pattern recognition, resource prediction, constraint adaptation, and recovery
management for optimal system performance.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time

from src.mcp.predictive_resource_allocation import (
    PredictiveResourceAllocation,
    ResourceMetrics,
    ResourceConstraints,
    AdaptationPlan,
    RecoveryState
)
from src.mcp.resource_optimization_engine import ResourceOptimizationEngine
from src.mcp.brain_state_aggregator import BrainStateAggregator
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
from src.mcp.workload_pattern_analyzer import WorkloadPattern, ResourcePrediction


class PredictiveResourceAllocationIntegration:
    """
    Integration of predictive resource allocation with resource optimization.
    
    This class integrates the PredictiveResourceAllocation system with the ResourceOptimizationEngine
    to provide workload pattern recognition, resource prediction, constraint adaptation, and recovery
    management for optimal system performance.
    """
    
    def __init__(
        self,
        resource_optimization_engine: Optional[ResourceOptimizationEngine] = None,
        brain_state_aggregator: Optional[BrainStateAggregator] = None,
        event_bus: Optional[LobeEventBus] = None,
        hormone_controller: Any = None
    ):
        """
        Initialize the predictive resource allocation integration.
        
        Args:
            resource_optimization_engine: Resource optimization engine
            brain_state_aggregator: Brain state aggregator
            event_bus: Event bus for emitting and receiving events
            hormone_controller: Hormone system controller
        """
        self.logger = logging.getLogger("PredictiveResourceAllocationIntegration")
        
        # Store dependencies
        self.resource_optimization_engine = resource_optimization_engine
        self.brain_state_aggregator = brain_state_aggregator
        self.event_bus = event_bus
        self.hormone_controller = hormone_controller
        
        # Create predictive resource allocation system
        self.predictive_allocation = PredictiveResourceAllocation(
            hormone_controller=hormone_controller,
            brain_state_aggregator=brain_state_aggregator,
            event_bus=event_bus
        )
        
        # Integration state
        self.integration_active = False
        self.update_interval = 10  # Update every 10 seconds
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Prediction state
        self.last_prediction_time = datetime.now() - timedelta(minutes=5)
        self.prediction_interval = 60  # Generate new predictions every 60 seconds
        self.current_prediction = None
        
        # Adaptation state
        self.last_adaptation_time = datetime.now() - timedelta(minutes=5)
        self.adaptation_interval = 30  # Apply adaptations every 30 seconds
        self.current_adaptation_plan = None
        
        # Recovery state
        self.in_recovery_mode = False
        self.recovery_start_time = None
        self.recovery_end_time = None
        
        # Set up event handlers
        self._setup_event_handlers()
        
        self.logger.info("PredictiveResourceAllocationIntegration initialized")
    
    def _setup_event_handlers(self):
        """Set up event handlers for resource-related events."""
        if self.event_bus:
            self.event_bus.subscribe("resource_constraint_detected", self._handle_resource_constraint)
            self.event_bus.subscribe("memory_pressure_detected", self._handle_memory_pressure)
            self.event_bus.subscribe("system_idle_detected", self._handle_system_idle)
            self.event_bus.subscribe("high_activity_detected", self._handle_high_activity)
            self.event_bus.subscribe("resource_recovery_detected", self._handle_resource_recovery)
            
            self.logger.info("Resource event handlers registered")
    
    def start(self):
        """Start the predictive resource allocation integration."""
        if self.integration_active:
            return
        
        self.integration_active = True
        self.stop_event.clear()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("PredictiveResourceAllocationIntegration started")
    
    def stop(self):
        """Stop the predictive resource allocation integration."""
        if not self.integration_active:
            return
        
        self.integration_active = False
        self.stop_event.set()
        
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
            self.update_thread = None
        
        self.logger.info("PredictiveResourceAllocationIntegration stopped")
    
    def _update_loop(self):
        """Main update loop for predictive resource allocation."""
        while not self.stop_event.is_set():
            try:
                # Update predictive allocation
                self.predictive_allocation.update()
                
                # Check if it's time to generate a new prediction
                if (datetime.now() - self.last_prediction_time).total_seconds() > self.prediction_interval:
                    self._generate_resource_prediction()
                    self.last_prediction_time = datetime.now()
                
                # Check if it's time to apply adaptations
                if (datetime.now() - self.last_adaptation_time).total_seconds() > self.adaptation_interval:
                    self._apply_predictive_adaptations()
                    self.last_adaptation_time = datetime.now()
                
                # Evaluate adaptation effectiveness
                self.predictive_allocation.evaluate_adaptation_effectiveness()
                
            except Exception as e:
                self.logger.error(f"Error in predictive resource allocation update: {e}")
            
            # Wait for next update
            self.stop_event.wait(self.update_interval)
    
    def _generate_resource_prediction(self):
        """Generate resource prediction and update the resource optimization engine."""
        # Get current prediction from predictive allocation
        prediction = self.predictive_allocation.get_current_prediction()
        if not prediction:
            return
        
        self.current_prediction = prediction
        
        # Log prediction
        self.logger.info(f"Generated resource prediction with {prediction.confidence:.2f} confidence")
        
        # Update brain state aggregator with prediction
        if self.brain_state_aggregator:
            environment_state = self.brain_state_aggregator.get_environment_state()
            if isinstance(environment_state, dict):
                # Add prediction to environment state
                environment_state["resource_prediction"] = {
                    "confidence": prediction.confidence,
                    "cpu_trend": "increasing" if any(v > self.predictive_allocation.current_metrics.cpu_usage + 5 
                                                  for v in prediction.predicted_cpu.values()) else "stable",
                    "memory_trend": "increasing" if any(v > self.predictive_allocation.current_metrics.memory_usage + 5 
                                                     for v in prediction.predicted_memory.values()) else "stable",
                    "recommended_actions": prediction.recommended_actions
                }
                
                # Update brain state
                if hasattr(self.brain_state_aggregator, "update_environment_state"):
                    self.brain_state_aggregator.update_environment_state(environment_state)
        
        # Apply predictive optimizations if confidence is high enough
        if prediction.confidence >= 0.6:
            self._apply_predictive_optimizations(prediction)
    
    def _apply_predictive_optimizations(self, prediction: ResourcePrediction):
        """
        Apply optimizations based on resource prediction.
        
        Args:
            prediction: Resource prediction
        """
        # Check for predicted high CPU usage
        max_predicted_cpu = max(prediction.predicted_cpu.values()) if prediction.predicted_cpu else 0
        cpu_threshold = 70.0  # 70% CPU usage triggers optimization
        
        if max_predicted_cpu > cpu_threshold:
            self.logger.info(f"Applying predictive CPU optimization (predicted: {max_predicted_cpu:.1f}%)")
            
            # Create CPU optimization plan
            if self.resource_optimization_engine:
                # Trigger workload pattern recognition
                if hasattr(self.resource_optimization_engine, "predict_workload_pattern"):
                    pattern = self.resource_optimization_engine.predict_workload_pattern()
                    
                    # Predict resource needs
                    if pattern and hasattr(self.resource_optimization_engine, "predict_resource_needs"):
                        resource_prediction = self.resource_optimization_engine.predict_resource_needs(pattern)
                        
                        # Apply predictive optimizations
                        if resource_prediction and hasattr(self.resource_optimization_engine, "_apply_predictive_optimizations"):
                            self.resource_optimization_engine._apply_predictive_optimizations(resource_prediction)
        
        # Check for predicted memory pressure
        max_predicted_memory = max(prediction.predicted_memory.values()) if prediction.predicted_memory else 0
        memory_threshold = 75.0  # 75% memory usage triggers optimization
        
        if max_predicted_memory > memory_threshold:
            self.logger.info(f"Applying predictive memory optimization (predicted: {max_predicted_memory:.1f}%)")
            
            # Trigger memory consolidation if available
            if self.resource_optimization_engine and hasattr(self.resource_optimization_engine, "trigger_memory_consolidation"):
                severity = (max_predicted_memory - memory_threshold) / (100.0 - memory_threshold)
                self.resource_optimization_engine.trigger_memory_consolidation(
                    max_predicted_memory / 100.0,
                    memory_threshold / 100.0
                )
    
    def _apply_predictive_adaptations(self):
        """Apply predictive adaptations based on current system state and predictions."""
        # Skip if no prediction available
        if not self.current_prediction:
            return
        
        # Skip if prediction confidence is too low
        if self.current_prediction.confidence < 0.5:
            return
        
        # Get current metrics
        current_metrics = self.predictive_allocation.current_metrics
        
        # Create constraints based on predicted resource usage
        constraints = ResourceConstraints(
            max_cpu_usage=80.0,  # 80% CPU usage threshold
            max_memory_usage=80.0,  # 80% memory usage threshold
            max_disk_usage=90.0,  # 90% disk usage threshold
            priority_lobes=[]  # No priority lobes by default
        )
        
        # Adjust constraints based on prediction
        max_predicted_cpu = max(self.current_prediction.predicted_cpu.values()) if self.current_prediction.predicted_cpu else 0
        max_predicted_memory = max(self.current_prediction.predicted_memory.values()) if self.current_prediction.predicted_memory else 0
        
        # If high CPU usage predicted, lower the threshold to trigger earlier adaptation
        if max_predicted_cpu > 70.0:
            constraints.max_cpu_usage = 70.0
        
        # If high memory usage predicted, lower the threshold to trigger earlier adaptation
        if max_predicted_memory > 70.0:
            constraints.max_memory_usage = 70.0
        
        # Determine priority lobes based on workload pattern
        current_pattern = self.predictive_allocation.pattern_analyzer.get_current_pattern()
        if current_pattern:
            if current_pattern.pattern_type == "cyclic":
                # For cyclic patterns, prioritize pattern recognition and scientific process
                constraints.priority_lobes = ["pattern_recognition", "scientific_process"]
            elif current_pattern.pattern_type == "bursty":
                # For bursty patterns, prioritize memory and task management
                constraints.priority_lobes = ["memory", "task_management"]
            else:  # steady
                # For steady patterns, balanced priorities
                constraints.priority_lobes = ["context_management", "decision_making"]
        
        # Generate adaptation plan
        adaptation_plan = self.predictive_allocation.adapt_to_resource_constraints(constraints)
        
        # Store current adaptation plan
        self.current_adaptation_plan = adaptation_plan
        
        # Apply adaptation plan to resource optimization engine
        if self.resource_optimization_engine:
            # Apply hormone adjustments
            if hasattr(self.resource_optimization_engine, "_apply_hormone_adjustments"):
                for hormone, level in adaptation_plan.hormone_adjustments.items():
                    if self.hormone_controller:
                        self.hormone_controller.set_hormone_level(hormone, level)
            
            # Apply memory consolidation
            if adaptation_plan.memory_consolidation_targets and hasattr(self.resource_optimization_engine, "trigger_memory_consolidation"):
                self.resource_optimization_engine.trigger_memory_consolidation(
                    current_metrics.memory_usage / 100.0,
                    0.7  # 70% threshold
                )
            
            # Apply background task adjustments
            if hasattr(self.resource_optimization_engine, "schedule_background_training"):
                # Create resource metrics for background training
                available_resources = ResourceMetrics(
                    cpu_usage=current_metrics.cpu_usage,
                    memory_usage=current_metrics.memory_usage,
                    memory_available=current_metrics.memory_available,
                    disk_usage=current_metrics.disk_usage
                )
                
                # Schedule background training with available resources
                self.resource_optimization_engine.schedule_background_training(available_resources)
    
    def _handle_resource_constraint(self, data: Dict[str, Any]):
        """
        Handle resource constraint event.
        
        Args:
            data: Event data containing constraint information
        """
        # Extract constraint information
        constraint_type = data.get("constraint_type", "unknown")
        severity = data.get("severity", 0.5)
        
        # Create constraint object
        constraints = ResourceConstraints(
            max_cpu_usage=data.get("max_cpu_usage", 80.0),
            max_memory_usage=data.get("max_memory_usage", 80.0),
            max_disk_usage=data.get("max_disk_usage", 90.0),
            priority_lobes=data.get("priority_lobes", [])
        )
        
        # Generate adaptation plan
        adaptation_plan = self.predictive_allocation.adapt_to_resource_constraints(constraints)
        
        # Apply adaptation plan
        if self.resource_optimization_engine and hasattr(self.resource_optimization_engine, "_apply_adaptation_plan"):
            self.resource_optimization_engine._apply_adaptation_plan(adaptation_plan)
        
        self.logger.info(f"Handled resource constraint event: {constraint_type} (severity: {severity:.2f})")
    
    def _handle_memory_pressure(self, data: Dict[str, Any]):
        """
        Handle memory pressure event.
        
        Args:
            data: Event data containing memory pressure information
        """
        # Extract memory pressure information
        memory_usage = data.get("memory_usage", 0.0)
        threshold = data.get("threshold", 0.7)
        
        # Trigger memory consolidation
        if self.resource_optimization_engine and hasattr(self.resource_optimization_engine, "trigger_memory_consolidation"):
            self.resource_optimization_engine.trigger_memory_consolidation(memory_usage, threshold)
        
        self.logger.info(f"Handled memory pressure event: {memory_usage:.1f}% (threshold: {threshold:.1f}%)")
    
    def _handle_system_idle(self, data: Dict[str, Any]):
        """
        Handle system idle event.
        
        Args:
            data: Event data containing idle information
        """
        # Extract available resources
        available_resources = ResourceMetrics(
            cpu_usage=data.get("cpu_usage", 0.0),
            memory_usage=data.get("memory_usage", 0.0),
            memory_available=data.get("memory_available", 0),
            disk_usage=data.get("disk_usage", 0.0)
        )
        
        # Schedule background training
        if self.resource_optimization_engine and hasattr(self.resource_optimization_engine, "schedule_background_training"):
            self.resource_optimization_engine.schedule_background_training(available_resources)
        
        self.logger.info(f"Handled system idle event: CPU {available_resources.cpu_usage:.1f}% available")
    
    def _handle_high_activity(self, data: Dict[str, Any]):
        """
        Handle high activity event.
        
        Args:
            data: Event data containing high activity information
        """
        # Extract active lobes
        active_lobes = data.get("active_lobes", [])
        resource_allocation = data.get("resource_allocation", {})
        
        # Prioritize resources for active lobes
        if self.resource_optimization_engine and hasattr(self.resource_optimization_engine, "prioritize_lobe_resources"):
            updated_allocation = self.resource_optimization_engine.prioritize_lobe_resources(
                active_lobes,
                resource_allocation
            )
            
            self.logger.info(f"Prioritized resources for active lobes: {', '.join(active_lobes)}")
    
    def _handle_resource_recovery(self, data: Dict[str, Any]):
        """
        Handle resource recovery event.
        
        Args:
            data: Event data containing recovery information
        """
        # Extract recovery information
        previous_hormone_levels = data.get("previous_hormone_levels", {})
        optimal_hormone_levels = data.get("optimal_hormone_levels", {})
        
        # Create recovery state
        recovery_state = RecoveryState(
            previous_hormone_levels=previous_hormone_levels,
            optimal_hormone_levels=optimal_hormone_levels
        )
        
        # Start recovery process
        if self.resource_optimization_engine and hasattr(self.resource_optimization_engine, "restore_optimal_levels"):
            self.resource_optimization_engine.restore_optimal_levels(recovery_state)
            
            # Update recovery state
            self.in_recovery_mode = True
            self.recovery_start_time = datetime.now()
            self.recovery_end_time = datetime.now() + timedelta(seconds=recovery_state.recovery_duration)
            
            self.logger.info(f"Started resource recovery process (duration: {recovery_state.recovery_duration:.1f}s)")
    
    def get_current_prediction(self) -> Optional[ResourcePrediction]:
        """
        Get the current resource prediction.
        
        Returns:
            ResourcePrediction object or None if no prediction available
        """
        return self.current_prediction
    
    def get_current_adaptation_plan(self) -> Optional[AdaptationPlan]:
        """
        Get the current adaptation plan.
        
        Returns:
            AdaptationPlan object or None if no plan available
        """
        return self.current_adaptation_plan
    
    def is_in_recovery_mode(self) -> bool:
        """
        Check if the system is in recovery mode.
        
        Returns:
            True if in recovery mode, False otherwise
        """
        return self.in_recovery_mode
    
    def get_recovery_progress(self) -> float:
        """
        Get the recovery progress.
        
        Returns:
            Recovery progress (0-1) or 0.0 if not in recovery mode
        """
        if not self.in_recovery_mode or not self.recovery_start_time or not self.recovery_end_time:
            return 0.0
        
        total_duration = (self.recovery_end_time - self.recovery_start_time).total_seconds()
        elapsed = (datetime.now() - self.recovery_start_time).total_seconds()
        
        return min(1.0, elapsed / total_duration)