"""
ImplementationSwitchingMonitor: Monitors and manages switching between algorithmic and neural implementations.

This module provides a monitoring system for tracking the performance of algorithmic and neural
implementations, managing automatic switching between them, and handling fallback mechanisms
when neural networks fail.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import os
import json

from src.mcp.brain_state_aggregator import BrainStateAggregator


@dataclass
class ImplementationPerformance:
    """Performance metrics for an implementation."""
    accuracy: float = 0.0  # Accuracy (0-1)
    latency: float = 0.0  # Processing time in ms
    resource_usage: float = 0.0  # Resource usage (0-1)
    error_rate: float = 0.0  # Error rate (0-1)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    failure_count: int = 0  # Number of failures
    success_count: int = 0  # Number of successful executions


@dataclass
class SwitchingThresholds:
    """Thresholds for implementation switching."""
    accuracy_threshold: float = 0.05  # Minimum accuracy improvement to switch
    latency_threshold: float = 0.1  # Minimum latency improvement to switch
    resource_threshold: float = 0.1  # Minimum resource usage improvement to switch
    error_threshold: float = 0.01  # Maximum error rate increase allowed
    failure_threshold: int = 3  # Number of failures before fallback
    stability_period: int = 60  # Seconds to wait before switching again


class ImplementationSwitchingMonitor:
    """
    Monitors and manages switching between algorithmic and neural implementations.
    
    This class tracks the performance of algorithmic and neural implementations,
    manages automatic switching between them, and handles fallback mechanisms
    when neural networks fail.
    """
    
    def __init__(self, brain_state_aggregator: Optional[BrainStateAggregator] = None):
        """
        Initialize the implementation switching monitor.
        
        Args:
            brain_state_aggregator: Brain state aggregator for reporting performance metrics
        """
        self.logger = logging.getLogger("ImplementationSwitchingMonitor")
        self.brain_state_aggregator = brain_state_aggregator
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, ImplementationPerformance]] = {}
        self.active_implementations: Dict[str, str] = {}
        self.last_switch_time: Dict[str, datetime] = {}
        self.switching_thresholds: Dict[str, SwitchingThresholds] = {}
        
        # Fallback state
        self.fallback_active: Dict[str, bool] = {}
        self.fallback_until: Dict[str, datetime] = {}
        
        # Model storage
        self.model_save_path = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.model_save_path, exist_ok=True)
        
        self.logger.info("ImplementationSwitchingMonitor initialized")
    
    def register_implementation(self, component: str, implementation_type: str):
        """
        Register an implementation for monitoring.
        
        Args:
            component: Name of the component
            implementation_type: Type of implementation ('algorithmic' or 'neural')
        """
        if component not in self.performance_metrics:
            self.performance_metrics[component] = {}
            self.switching_thresholds[component] = SwitchingThresholds()
        
        if implementation_type not in self.performance_metrics[component]:
            self.performance_metrics[component][implementation_type] = ImplementationPerformance()
        
        # Set as active implementation if none is active
        if component not in self.active_implementations:
            self.active_implementations[component] = implementation_type
            self.last_switch_time[component] = datetime.now()
            self.fallback_active[component] = False
        
        self.logger.info(f"Registered {implementation_type} implementation for {component}")
    
    def record_performance(self, component: str, implementation_type: str, 
                          metrics: Dict[str, float], success: bool = True):
        """
        Record performance metrics for an implementation.
        
        Args:
            component: Name of the component
            implementation_type: Type of implementation ('algorithmic' or 'neural')
            metrics: Dictionary of performance metrics
            success: Whether the execution was successful
        """
        # Ensure component and implementation are registered
        if component not in self.performance_metrics:
            self.register_implementation(component, implementation_type)
        elif implementation_type not in self.performance_metrics[component]:
            self.performance_metrics[component][implementation_type] = ImplementationPerformance()
        
        # Update performance metrics
        performance = self.performance_metrics[component][implementation_type]
        
        # Update metrics
        if "accuracy" in metrics:
            performance.accuracy = metrics["accuracy"]
        if "latency" in metrics:
            performance.latency = metrics["latency"]
        if "resource_usage" in metrics:
            performance.resource_usage = metrics["resource_usage"]
        if "error_rate" in metrics:
            performance.error_rate = metrics["error_rate"]
        
        # Update success/failure counts
        if success:
            performance.success_count += 1
        else:
            performance.failure_count += 1
        
        # Update timestamp
        performance.last_updated = datetime.now().isoformat()
        
        # Report to brain state aggregator
        if self.brain_state_aggregator:
            self.brain_state_aggregator.register_implementation_performance(
                component, implementation_type, metrics
            )
        
        # Check for failures that might trigger fallback
        if not success and implementation_type == "neural":
            self._check_fallback_needed(component)
        
        # Check if implementation switch is needed
        self._check_implementation_switch(component)
        
        self.logger.debug(f"Recorded {implementation_type} performance for {component}: "
                        f"{', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])}")
    
    def _check_fallback_needed(self, component: str):
        """
        Check if fallback to algorithmic implementation is needed.
        
        Args:
            component: Name of the component
        """
        if component not in self.performance_metrics:
            return
        
        # Check if neural implementation exists and is active
        if ("neural" not in self.performance_metrics[component] or
            self.active_implementations.get(component) != "neural"):
            return
        
        # Get neural performance
        neural_perf = self.performance_metrics[component]["neural"]
        
        # Check failure threshold
        thresholds = self.switching_thresholds[component]
        if neural_perf.failure_count >= thresholds.failure_threshold:
            self._activate_fallback(component)
    
    def _activate_fallback(self, component: str):
        """
        Activate fallback to algorithmic implementation.
        
        Args:
            component: Name of the component
        """
        # Switch to algorithmic implementation
        if "algorithmic" in self.performance_metrics[component]:
            self.active_implementations[component] = "algorithmic"
            self.last_switch_time[component] = datetime.now()
            
            # Set fallback state
            self.fallback_active[component] = True
            self.fallback_until[component] = datetime.now() + timedelta(minutes=5)
            
            # Reset failure count
            if "neural" in self.performance_metrics[component]:
                self.performance_metrics[component]["neural"].failure_count = 0
            
            self.logger.warning(f"Activated fallback to algorithmic implementation for {component} "
                              f"due to neural implementation failures")
            
            # Report to brain state aggregator
            if self.brain_state_aggregator:
                self.brain_state_aggregator._switch_implementation(
                    component, "algorithmic", datetime.now().isoformat()
                )
    
    def _check_implementation_switch(self, component: str):
        """
        Check if implementation switch is needed based on performance metrics.
        
        Args:
            component: Name of the component
        """
        if component not in self.performance_metrics:
            return
        
        # Check if both implementations exist
        if ("algorithmic" not in self.performance_metrics[component] or
            "neural" not in self.performance_metrics[component]):
            return
        
        # Check if fallback is active
        if self.fallback_active.get(component, False):
            if datetime.now() < self.fallback_until.get(component, datetime.now()):
                return  # Still in fallback period
            else:
                # End fallback period
                self.fallback_active[component] = False
        
        # Check stability period
        if component in self.last_switch_time:
            elapsed = (datetime.now() - self.last_switch_time[component]).total_seconds()
            if elapsed < self.switching_thresholds[component].stability_period:
                return  # Too soon to switch again
        
        # Get performance metrics
        algo_perf = self.performance_metrics[component]["algorithmic"]
        neural_perf = self.performance_metrics[component]["neural"]
        
        # Get current active implementation
        current_impl = self.active_implementations.get(component, "algorithmic")
        
        # Calculate improvement scores
        thresholds = self.switching_thresholds[component]
        
        # Higher accuracy is better
        accuracy_improvement = neural_perf.accuracy - algo_perf.accuracy
        
        # Lower latency is better
        latency_improvement = (algo_perf.latency - neural_perf.latency) / max(1.0, algo_perf.latency)
        
        # Lower resource usage is better
        resource_improvement = (algo_perf.resource_usage - neural_perf.resource_usage) / max(1.0, algo_perf.resource_usage)
        
        # Lower error rate is better
        error_difference = neural_perf.error_rate - algo_perf.error_rate
        
        # Decide whether to switch
        switch_to_neural = (
            accuracy_improvement > thresholds.accuracy_threshold and
            latency_improvement > -thresholds.latency_threshold and  # Allow slightly worse latency
            resource_improvement > -thresholds.resource_threshold and  # Allow slightly worse resource usage
            error_difference < thresholds.error_threshold and
            neural_perf.failure_count < thresholds.failure_threshold and
            current_impl != "neural"
        )
        
        switch_to_algorithmic = (
            accuracy_improvement < -thresholds.accuracy_threshold or
            latency_improvement < -thresholds.latency_threshold or
            resource_improvement < -thresholds.resource_threshold or
            error_difference > thresholds.error_threshold or
            neural_perf.failure_count >= thresholds.failure_threshold
        ) and current_impl != "algorithmic"
        
        # Perform switch if needed
        if switch_to_neural:
            self._switch_implementation(component, "neural")
        elif switch_to_algorithmic:
            self._switch_implementation(component, "algorithmic")
    
    def _switch_implementation(self, component: str, implementation_type: str):
        """
        Switch the active implementation for a component.
        
        Args:
            component: Name of the component
            implementation_type: Type of implementation to switch to
        """
        old_impl = self.active_implementations.get(component, "unknown")
        self.active_implementations[component] = implementation_type
        self.last_switch_time[component] = datetime.now()
        
        # Reset fallback state if switching to neural
        if implementation_type == "neural":
            self.fallback_active[component] = False
        
        self.logger.info(f"Switched {component} implementation from {old_impl} to {implementation_type}")
        
        # Report to brain state aggregator
        if self.brain_state_aggregator:
            self.brain_state_aggregator._switch_implementation(
                component, implementation_type, datetime.now().isoformat()
            )
    
    def get_active_implementation(self, component: str) -> str:
        """
        Get the currently active implementation for a component.
        
        Args:
            component: Name of the component
            
        Returns:
            Name of the active implementation, or 'algorithmic' if not found
        """
        return self.active_implementations.get(component, "algorithmic")
    
    def save_improved_model(self, component: str, model_data: Any):
        """
        Save an improved neural model for future use.
        
        Args:
            component: Name of the component
            model_data: Model data to save
        """
        try:
            # Create component directory if it doesn't exist
            component_dir = os.path.join(self.model_save_path, component)
            os.makedirs(component_dir, exist_ok=True)
            
            # Save model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(component_dir, f"model_{timestamp}.json")
            
            # For this example, we'll save as JSON, but in practice this would be a model format
            with open(model_path, 'w') as f:
                if isinstance(model_data, dict):
                    json.dump(model_data, f)
                else:
                    json.dump({"model_data": str(model_data)}, f)
            
            # Save performance metrics
            if component in self.performance_metrics and "neural" in self.performance_metrics[component]:
                metrics_path = os.path.join(component_dir, f"metrics_{timestamp}.json")
                with open(metrics_path, 'w') as f:
                    json.dump(vars(self.performance_metrics[component]["neural"]), f)
            
            self.logger.info(f"Saved improved model for {component} to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model for {component}: {e}")
            return None
    
    def load_model(self, component: str) -> Tuple[Any, bool]:
        """
        Load the latest model for a component.
        
        Args:
            component: Name of the component
            
        Returns:
            Tuple of (model_data, success)
        """
        try:
            # Check if component directory exists
            component_dir = os.path.join(self.model_save_path, component)
            if not os.path.exists(component_dir):
                return None, False
            
            # Find the latest model file
            model_files = [f for f in os.listdir(component_dir) if f.startswith("model_")]
            if not model_files:
                return None, False
            
            # Sort by timestamp (newest first)
            model_files.sort(reverse=True)
            latest_model = os.path.join(component_dir, model_files[0])
            
            # Load model
            with open(latest_model, 'r') as f:
                model_data = json.load(f)
            
            self.logger.info(f"Loaded model for {component} from {latest_model}")
            return model_data, True
            
        except Exception as e:
            self.logger.error(f"Error loading model for {component}: {e}")
            return None, False
    
    def set_switching_thresholds(self, component: str, thresholds: Dict[str, float]):
        """
        Set thresholds for implementation switching.
        
        Args:
            component: Name of the component
            thresholds: Dictionary of threshold values
        """
        if component not in self.switching_thresholds:
            self.switching_thresholds[component] = SwitchingThresholds()
        
        # Update thresholds
        if "accuracy_threshold" in thresholds:
            self.switching_thresholds[component].accuracy_threshold = thresholds["accuracy_threshold"]
        if "latency_threshold" in thresholds:
            self.switching_thresholds[component].latency_threshold = thresholds["latency_threshold"]
        if "resource_threshold" in thresholds:
            self.switching_thresholds[component].resource_threshold = thresholds["resource_threshold"]
        if "error_threshold" in thresholds:
            self.switching_thresholds[component].error_threshold = thresholds["error_threshold"]
        if "failure_threshold" in thresholds:
            self.switching_thresholds[component].failure_threshold = thresholds["failure_threshold"]
        if "stability_period" in thresholds:
            self.switching_thresholds[component].stability_period = thresholds["stability_period"]
        
        self.logger.info(f"Updated switching thresholds for {component}")
    
    def execute_with_fallback(self, component: str, neural_func: Callable, 
                             algorithmic_func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with automatic implementation selection and fallback.
        
        Args:
            component: Name of the component
            neural_func: Neural implementation function
            algorithmic_func: Algorithmic implementation function
            *args, **kwargs: Arguments to pass to the functions
            
        Returns:
            Result of the selected implementation
        """
        # Get active implementation
        active_impl = self.get_active_implementation(component)
        
        # Select function based on active implementation
        primary_func = neural_func if active_impl == "neural" else algorithmic_func
        fallback_func = algorithmic_func if active_impl == "neural" else None
        
        # Try primary implementation
        start_time = time.time()
        try:
            result = primary_func(*args, **kwargs)
            
            # Record successful execution
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self.record_performance(
                component, active_impl, 
                {"latency": execution_time, "error_rate": 0.0},
                success=True
            )
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self.record_performance(
                component, active_impl, 
                {"latency": execution_time, "error_rate": 1.0},
                success=False
            )
            
            self.logger.warning(f"Error in {active_impl} implementation of {component}: {e}")
            
            # Try fallback if available
            if fallback_func:
                self.logger.info(f"Falling back to algorithmic implementation for {component}")
                
                start_time = time.time()
                try:
                    result = fallback_func(*args, **kwargs)
                    
                    # Record successful fallback
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    self.record_performance(
                        component, "algorithmic", 
                        {"latency": execution_time, "error_rate": 0.0},
                        success=True
                    )
                    
                    return result
                    
                except Exception as fallback_error:
                    # Record fallback failure
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    self.record_performance(
                        component, "algorithmic", 
                        {"latency": execution_time, "error_rate": 1.0},
                        success=False
                    )
                    
                    self.logger.error(f"Error in fallback implementation of {component}: {fallback_error}")
                    raise fallback_error
            else:
                # No fallback available, re-raise original exception
                raise e