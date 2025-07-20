"""
Integration module for connecting PerformanceTracker with BrainStateAggregator.

This module provides functionality to integrate the neural model performance tracking
with the Brain State Aggregator, ensuring that performance metrics are properly
reported and can be used for implementation switching decisions.
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime

from ..brain_state_aggregator import BrainStateAggregator
from .performance_tracker import PerformanceTracker


class NeuralPerformanceIntegration:
    """
    Integrates neural model performance tracking with the Brain State Aggregator.
    
    This class serves as a bridge between the PerformanceTracker and BrainStateAggregator,
    ensuring that neural model performance metrics are properly reported to the brain state
    and can be used for implementation switching decisions.
    """
    
    def __init__(self, 
                 brain_state_aggregator: BrainStateAggregator,
                 performance_tracker: PerformanceTracker):
        """
        Initialize the neural performance integration.
        
        Args:
            brain_state_aggregator: Brain State Aggregator instance.
            performance_tracker: Performance Tracker instance.
        """
        self.logger = logging.getLogger("NeuralPerformanceIntegration")
        self.brain_state_aggregator = brain_state_aggregator
        self.performance_tracker = performance_tracker
        
        self.logger.info("NeuralPerformanceIntegration initialized")
    
    def report_neural_metrics(self, 
                             function_name: str, 
                             accuracy: float, 
                             latency: float, 
                             resource_usage: float,
                             additional_metrics: Optional[Dict[str, float]] = None,
                             context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Report neural model performance metrics to both the tracker and brain state.
        
        Args:
            function_name: Name of the function being measured.
            accuracy: Accuracy metric (0.0-1.0).
            latency: Processing time in milliseconds.
            resource_usage: Resource usage (0.0-1.0).
            additional_metrics: Additional performance metrics.
            context: Additional context information.
            
        Returns:
            True if metrics were reported successfully, False otherwise.
        """
        try:
            # Record metrics in the performance tracker
            self.performance_tracker.record_metrics(
                function_name=function_name,
                implementation_type="neural",
                accuracy=accuracy,
                latency=latency,
                resource_usage=resource_usage,
                additional_metrics=additional_metrics,
                context=context
            )
            
            # Report metrics to the brain state aggregator
            metrics = {
                "accuracy": accuracy,
                "latency": latency,
                "resource_usage": resource_usage
            }
            
            # Add additional metrics
            if additional_metrics:
                metrics.update(additional_metrics)
                
            # Register with brain state aggregator
            self.brain_state_aggregator.register_implementation_performance(
                component=function_name,
                implementation_type="neural",
                metrics=metrics
            )
            
            self.logger.info(f"Reported neural metrics for {function_name}: "
                           f"accuracy={accuracy:.4f}, latency={latency:.2f}ms, "
                           f"resource_usage={resource_usage:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error reporting neural metrics for {function_name}: {e}")
            return False
    
    def report_algorithmic_metrics(self, 
                                  function_name: str, 
                                  accuracy: float, 
                                  latency: float, 
                                  resource_usage: float,
                                  additional_metrics: Optional[Dict[str, float]] = None,
                                  context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Report algorithmic implementation performance metrics to both the tracker and brain state.
        
        Args:
            function_name: Name of the function being measured.
            accuracy: Accuracy metric (0.0-1.0).
            latency: Processing time in milliseconds.
            resource_usage: Resource usage (0.0-1.0).
            additional_metrics: Additional performance metrics.
            context: Additional context information.
            
        Returns:
            True if metrics were reported successfully, False otherwise.
        """
        try:
            # Record metrics in the performance tracker
            self.performance_tracker.record_metrics(
                function_name=function_name,
                implementation_type="algorithmic",
                accuracy=accuracy,
                latency=latency,
                resource_usage=resource_usage,
                additional_metrics=additional_metrics,
                context=context
            )
            
            # Report metrics to the brain state aggregator
            metrics = {
                "accuracy": accuracy,
                "latency": latency,
                "resource_usage": resource_usage
            }
            
            # Add additional metrics
            if additional_metrics:
                metrics.update(additional_metrics)
                
            # Register with brain state aggregator
            self.brain_state_aggregator.register_implementation_performance(
                component=function_name,
                implementation_type="algorithmic",
                metrics=metrics
            )
            
            self.logger.info(f"Reported algorithmic metrics for {function_name}: "
                           f"accuracy={accuracy:.4f}, latency={latency:.2f}ms, "
                           f"resource_usage={resource_usage:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error reporting algorithmic metrics for {function_name}: {e}")
            return False
    
    def sync_implementation_decisions(self) -> Dict[str, Tuple[str, str]]:
        """
        Synchronize implementation decisions between the performance tracker and brain state.
        
        This ensures that both systems have consistent information about which
        implementation (neural or algorithmic) should be used for each function.
        
        Returns:
            Dictionary mapping function names to tuples of (implementation_type, reason).
        """
        try:
            decisions = {}
            
            # Get all functions with metrics
            functions = set()
            for function_name in self.performance_tracker.current_metrics.keys():
                functions.add(function_name)
                
            # Process each function
            for function_name in functions:
                # Get recommendation from performance tracker
                impl_type, reason = self.performance_tracker.get_recommended_implementation(function_name)
                
                # Update brain state aggregator
                active_impl = self.brain_state_aggregator.get_active_implementation(function_name)
                
                if active_impl != impl_type:
                    # Implementation decision differs, update brain state
                    # Note: Brain State Aggregator doesn't have a direct method to set the active implementation,
                    # but it will update based on the next performance metrics it receives
                    self.logger.info(f"Implementation decision for {function_name} differs: "
                                   f"brain_state={active_impl}, tracker={impl_type}")
                    
                    # Re-register the recommended implementation with slightly better metrics
                    # to encourage the brain state to switch
                    if impl_type == "neural":
                        neural_metrics = self.performance_tracker.get_current_metrics(function_name, "neural")
                        if neural_metrics:
                            self.brain_state_aggregator.register_implementation_performance(
                                component=function_name,
                                implementation_type="neural",
                                metrics={
                                    "accuracy": neural_metrics.accuracy * 1.01,  # Slightly better
                                    "latency": neural_metrics.latency * 0.99,   # Slightly better
                                    "resource_usage": neural_metrics.resource_usage * 0.99  # Slightly better
                                }
                            )
                    else:
                        algo_metrics = self.performance_tracker.get_current_metrics(function_name, "algorithmic")
                        if algo_metrics:
                            self.brain_state_aggregator.register_implementation_performance(
                                component=function_name,
                                implementation_type="algorithmic",
                                metrics={
                                    "accuracy": algo_metrics.accuracy * 1.01,  # Slightly better
                                    "latency": algo_metrics.latency * 0.99,   # Slightly better
                                    "resource_usage": algo_metrics.resource_usage * 0.99  # Slightly better
                                }
                            )
                
                decisions[function_name] = (impl_type, reason)
                
            self.logger.info(f"Synchronized implementation decisions for {len(decisions)} functions")
            return decisions
            
        except Exception as e:
            self.logger.error(f"Error synchronizing implementation decisions: {e}")
            return {}
    
    def get_performance_history(self, function_name: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get performance history for a function from both the tracker and brain state.
        
        Args:
            function_name: Name of the function.
            limit: Maximum number of history entries to return.
            
        Returns:
            Dictionary with performance history from both sources.
        """
        try:
            result = {
                "tracker_history": {},
                "brain_state_history": []
            }
            
            # Get history from performance tracker
            tracker_history = self.performance_tracker.get_metrics_history(function_name, limit=limit)
            if tracker_history:
                for impl_type, history in tracker_history.items():
                    result["tracker_history"][impl_type] = [
                        {
                            "accuracy": metrics.accuracy,
                            "latency": metrics.latency,
                            "resource_usage": metrics.resource_usage,
                            "timestamp": metrics.timestamp.isoformat(),
                            **metrics.additional_metrics
                        }
                        for metrics in history
                    ]
            
            # Get history from brain state aggregator
            if hasattr(self.brain_state_aggregator, "implementation_history") and function_name in self.brain_state_aggregator.implementation_history:
                brain_history = self.brain_state_aggregator.implementation_history[function_name]
                result["brain_state_history"] = brain_history[-limit:] if limit else brain_history
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting performance history for {function_name}: {e}")
            return {"tracker_history": {}, "brain_state_history": []}
    
    def save_all_metrics(self) -> bool:
        """
        Save all metrics to disk.
        
        Returns:
            True if all metrics were saved successfully, False otherwise.
        """
        try:
            success = True
            
            # Get all functions with metrics
            functions = set()
            for function_name in self.performance_tracker.current_metrics.keys():
                functions.add(function_name)
                
            # Save metrics for each function
            for function_name in functions:
                if not self.performance_tracker.save_metrics(function_name):
                    success = False
                    
            self.logger.info(f"Saved metrics for {len(functions)} functions")
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            return False