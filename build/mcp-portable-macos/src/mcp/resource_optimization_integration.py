"""
ResourceOptimizationIntegration: Integration module for resource optimization components.

This module integrates the PredictiveResourceAllocation system with the existing
ResourceOptimizationEngine to provide advanced resource optimization capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta

from src.mcp.hormone_system_controller import HormoneSystemController
from src.mcp.brain_state_aggregator import BrainStateAggregator
from src.mcp.lobes.experimental.lobe_event_bus import LobeEventBus
from src.mcp.workload_pattern_analyzer import WorkloadPatternAnalyzer, WorkloadPattern, ResourcePrediction
from src.mcp.predictive_resource_allocation import (
    PredictiveResourceAllocation, ResourceMetrics, ResourceConstraints, 
    AdaptationPlan, RecoveryState
)
from src.mcp.implementation_switching_monitor import ImplementationSwitchingMonitor


class ResourceOptimizationIntegration:
    """
    Integration class for resource optimization components.
    
    This class integrates the PredictiveResourceAllocation system with the existing
    ResourceOptimizationEngine to provide advanced resource optimization capabilities.
    """
    
    def __init__(self, hormone_controller=None, brain_state_aggregator=None, event_bus=None):
        """
        Initialize the resource optimization integration.
        
        Args:
            hormone_controller: Hormone system controller for adjusting hormone production
            brain_state_aggregator: Brain state aggregator for monitoring system state
            event_bus: Event bus for emitting and receiving events
        """
        self.logger = logging.getLogger("ResourceOptimizationIntegration")
        
        # Store dependencies
        self.hormone_controller = hormone_controller
        self.brain_state_aggregator = brain_state_aggregator
        self.event_bus = event_bus or LobeEventBus()
        
        # Create components
        self.predictive_allocation = PredictiveResourceAllocation(
            hormone_controller=hormone_controller,
            brain_state_aggregator=brain_state_aggregator,
            event_bus=self.event_bus
        )
        
        self.implementation_monitor = ImplementationSwitchingMonitor(
            brain_state_aggregator=brain_state_aggregator
        )
        
        # Set up event handlers
        self._setup_event_handlers()
        
        self.logger.info("ResourceOptimizationIntegration initialized")
    
    def _setup_event_handlers(self):
        """Set up event handlers for resource-related events."""
        self.event_bus.subscribe("resource_constraint_detected", self._handle_resource_constraint)
        self.event_bus.subscribe("implementation_performance_updated", self._handle_implementation_performance)
        self.event_bus.subscribe("high_activity_detected", self._handle_high_activity)
        
        self.logger.info("Resource event handlers registered")
    
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
            max_network_usage=data.get("max_network_usage", 1000000),
            priority_lobes=data.get("priority_lobes", [])
        )
        
        # Generate adaptation plan
        adaptation_plan = self.predictive_allocation.adapt_to_resource_constraints(constraints)
        
        # Apply adaptation plan
        self.predictive_allocation._apply_adaptation_plan(adaptation_plan)
        
        self.logger.info(f"Adapted to {constraint_type} constraint (severity: {severity:.2f})")
    
    def _handle_implementation_performance(self, data: Dict[str, Any]):
        """
        Handle implementation performance update event.
        
        Args:
            data: Event data containing performance information
        """
        component = data.get("component")
        implementation_type = data.get("implementation_type")
        metrics = data.get("metrics", {})
        
        if component and implementation_type and metrics:
            self.implementation_monitor.register_implementation_performance(
                component, implementation_type, metrics
            )
    
    def _handle_high_activity(self, data: Dict[str, Any]):
        """
        Handle high activity event.
        
        Args:
            data: Event data containing activity information
        """
        # Extract active lobes and their activity levels
        active_lobes = data.get("active_lobes", {})
        
        # Prioritize resources for active lobes
        if active_lobes:
            resource_allocation = data.get("current_allocation", {})
            updated_allocation = self.predictive_allocation.prioritize_lobe_resources(
                list(active_lobes.keys()),
                resource_allocation
            )
            
            self.logger.info(f"Prioritized resources for {len(active_lobes)} active lobes")
            
            # Emit updated allocation event
            self.event_bus.emit(
                "resource_allocation_updated",
                {
                    "allocation": updated_allocation,
                    "active_lobes": active_lobes
                }
            )
    
    def update(self):
        """Update resource optimization state."""
        # Update predictive allocation
        self.predictive_allocation.update()
        
        # Check fallback status
        self.implementation_monitor.check_fallback_status()
        
        # Get current prediction
        prediction = self.predictive_allocation.get_current_prediction()
        if prediction and prediction.confidence > 0.7:
            # Log high-confidence predictions
            self.logger.info(f"High-confidence prediction: {prediction.confidence:.2f}")
            
            # Check for recommended actions
            if prediction.recommended_actions:
                self.logger.info(f"Recommended actions: {', '.join(prediction.recommended_actions)}")
    
    def get_active_implementation(self, component: str) -> str:
        """
        Get the currently active implementation for a component.
        
        Args:
            component: Name of the component
            
        Returns:
            Implementation type ('algorithmic' or 'neural')
        """
        return self.implementation_monitor.get_active_implementation(component)
    
    def get_current_prediction(self) -> Optional[ResourcePrediction]:
        """
        Get the current resource prediction.
        
        Returns:
            ResourcePrediction object or None if no prediction available
        """
        return self.predictive_allocation.get_current_prediction()
    
    def get_switching_history(self, component: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get implementation switching history for a component.
        
        Args:
            component: Name of the component
            limit: Maximum number of history entries to return
            
        Returns:
            List of switching history entries
        """
        return self.implementation_monitor.get_switching_history(component, limit)


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create integration
    integration = ResourceOptimizationIntegration()
    
    # Update a few times
    for _ in range(5):
        integration.update()
        
    # Test implementation switching
    integration.implementation_monitor.register_implementation_performance(
        "memory_system", "algorithmic", {"accuracy": 0.8, "speed": 50.0, "resource_usage": 0.3}
    )
    
    integration.implementation_monitor.register_implementation_performance(
        "memory_system", "neural", {"accuracy": 0.9, "speed": 40.0, "resource_usage": 0.5}
    )
    
    # Check active implementation
    active_impl = integration.get_active_implementation("memory_system")
    print(f"Active implementation for memory_system: {active_impl}")
    
    # Check switching history
    history = integration.get_switching_history("memory_system")
    if history:
        print("Implementation switching history:")
        for entry in history:
            print(f"  {entry['timestamp']}: {entry['from']} -> {entry['to']} ({entry['reason']})")