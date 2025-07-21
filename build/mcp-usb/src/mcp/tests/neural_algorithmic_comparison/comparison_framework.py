"""
Comparison Framework for Neural vs. Algorithmic Implementations.

This module provides a framework for comparing the performance of neural network
implementations against algorithmic implementations for hormone calculations.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class ImplementationPerformance:
    """
    Represents performance metrics for an implementation.
    
    This class stores performance metrics for either a neural or algorithmic
    implementation, including accuracy, latency, and resource usage.
    """
    
    def __init__(self, 
                 function_name: str,
                 implementation_type: str,
                 accuracy: float = 0.0,
                 latency: float = 0.0,
                 resource_usage: float = 0.0):
        """
        Initialize performance metrics.
        
        Args:
            function_name: Name of the function being implemented.
            implementation_type: Type of implementation ("neural" or "algorithmic").
            accuracy: Accuracy metric (0.0 to 1.0).
            latency: Processing time in milliseconds.
            resource_usage: Memory/CPU usage metric.
        """
        self.function_name = function_name
        self.implementation_type = implementation_type
        self.accuracy = accuracy
        self.latency = latency
        self.resource_usage = resource_usage
        self.timestamp = datetime.now()
        self.context: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert performance metrics to a dictionary.
        
        Returns:
            Dictionary representation of performance metrics.
        """
        return {
            "function_name": self.function_name,
            "implementation_type": self.implementation_type,
            "accuracy": self.accuracy,
            "latency": self.latency,
            "resource_usage": self.resource_usage,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImplementationPerformance':
        """
        Create performance metrics from a dictionary.
        
        Args:
            data: Dictionary representation of performance metrics.
            
        Returns:
            A new ImplementationPerformance instance.
        """
        performance = cls(
            function_name=data["function_name"],
            implementation_type=data["implementation_type"],
            accuracy=data["accuracy"],
            latency=data["latency"],
            resource_usage=data["resource_usage"]
        )
        
        performance.timestamp = datetime.fromisoformat(data["timestamp"])
        performance.context = data["context"]
        
        return performance


class ComparisonFramework:
    """
    Framework for comparing neural vs. algorithmic implementations.
    
    This class provides tools for measuring and comparing the performance of
    neural network implementations against algorithmic implementations.
    """
    
    def __init__(self):
        """Initialize the comparison framework."""
        self.logger = logging.getLogger("ComparisonFramework")
        self.performance_history: Dict[str, List[ImplementationPerformance]] = {}
        self.current_implementation: Dict[str, str] = {}  # function_name -> implementation_type
    
    def measure_performance(self, 
                           function_name: str,
                           neural_impl: Callable,
                           algo_impl: Callable,
                           test_data: List[Tuple[Any, Any]],
                           context: Dict[str, Any] = None) -> Dict[str, ImplementationPerformance]:
        """
        Measure performance of both implementations on test data.
        
        Args:
            function_name: Name of the function being tested.
            neural_impl: Neural network implementation function.
            algo_impl: Algorithmic implementation function.
            test_data: List of (input, expected_output) tuples.
            context: Additional context information.
            
        Returns:
            Dictionary with performance metrics for both implementations.
        """
        context = context or {}
        
        # Measure neural implementation performance
        neural_perf = self._measure_single_implementation(
            function_name, "neural", neural_impl, test_data
        )
        neural_perf.context = dict(context)
        
        # Measure algorithmic implementation performance
        algo_perf = self._measure_single_implementation(
            function_name, "algorithmic", algo_impl, test_data
        )
        algo_perf.context = dict(context)
        
        # Store performance history
        if function_name not in self.performance_history:
            self.performance_history[function_name] = []
        
        self.performance_history[function_name].append(neural_perf)
        self.performance_history[function_name].append(algo_perf)
        
        # Log performance comparison
        self.logger.info(f"Performance comparison for {function_name}:")
        self.logger.info(f"  Neural: accuracy={neural_perf.accuracy:.4f}, latency={neural_perf.latency:.4f}ms")
        self.logger.info(f"  Algorithmic: accuracy={algo_perf.accuracy:.4f}, latency={algo_perf.latency:.4f}ms")
        
        return {
            "neural": neural_perf,
            "algorithmic": algo_perf
        }
    
    def _measure_single_implementation(self,
                                      function_name: str,
                                      impl_type: str,
                                      implementation: Callable,
                                      test_data: List[Tuple[Any, Any]]) -> ImplementationPerformance:
        """
        Measure performance of a single implementation.
        
        Args:
            function_name: Name of the function being tested.
            impl_type: Implementation type ("neural" or "algorithmic").
            implementation: Implementation function.
            test_data: List of (input, expected_output) tuples.
            
        Returns:
            Performance metrics for the implementation.
        """
        if not test_data:
            return ImplementationPerformance(function_name, impl_type)
        
        correct = 0
        total_latency = 0
        total_memory = 0
        
        for input_data, expected_output in test_data:
            # Measure latency
            start_time = time.time()
            
            try:
                # Run implementation
                actual_output = implementation(input_data)
                
                # Check correctness
                if self._is_correct(actual_output, expected_output):
                    correct += 1
            except Exception as e:
                self.logger.error(f"Error in {impl_type} implementation of {function_name}: {e}")
            
            # Calculate latency in milliseconds
            latency = (time.time() - start_time) * 1000
            total_latency += latency
            
            # Placeholder for memory measurement
            # In a real implementation, this would use a library like psutil
            # to measure memory usage
            total_memory += 1.0  # Dummy value
        
        # Calculate average metrics
        accuracy = correct / len(test_data)
        avg_latency = total_latency / len(test_data)
        avg_memory = total_memory / len(test_data)
        
        return ImplementationPerformance(
            function_name=function_name,
            implementation_type=impl_type,
            accuracy=accuracy,
            latency=avg_latency,
            resource_usage=avg_memory
        )
    
    def _is_correct(self, actual: Any, expected: Any) -> bool:
        """
        Check if actual output matches expected output.
        
        Args:
            actual: Actual output from implementation.
            expected: Expected output.
            
        Returns:
            True if outputs match, False otherwise.
        """
        # Handle different types of outputs
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # For numeric values, allow small differences
            return abs(actual - expected) < 1e-6
        else:
            # For other types, require exact match
            return actual == expected
    
    def select_best_implementation(self, 
                                  function_name: str,
                                  accuracy_weight: float = 0.6,
                                  latency_weight: float = 0.3,
                                  resource_weight: float = 0.1) -> str:
        """
        Select the best implementation based on weighted performance metrics.
        
        Args:
            function_name: Name of the function to select implementation for.
            accuracy_weight: Weight for accuracy in selection (0.0 to 1.0).
            latency_weight: Weight for latency in selection (0.0 to 1.0).
            resource_weight: Weight for resource usage in selection (0.0 to 1.0).
            
        Returns:
            Implementation type ("neural" or "algorithmic").
        """
        if function_name not in self.performance_history:
            self.logger.warning(f"No performance history for {function_name}, defaulting to algorithmic")
            return "algorithmic"
        
        # Get latest performance metrics for each implementation
        neural_perf = None
        algo_perf = None
        
        for perf in reversed(self.performance_history[function_name]):
            if perf.implementation_type == "neural" and neural_perf is None:
                neural_perf = perf
            elif perf.implementation_type == "algorithmic" and algo_perf is None:
                algo_perf = perf
            
            if neural_perf and algo_perf:
                break
        
        if not neural_perf or not algo_perf:
            self.logger.warning(f"Incomplete performance data for {function_name}, defaulting to algorithmic")
            return "algorithmic"
        
        # Calculate weighted scores
        # For latency and resource usage, lower is better, so we use (1 - normalized_value)
        max_latency = max(neural_perf.latency, algo_perf.latency)
        max_resource = max(neural_perf.resource_usage, algo_perf.resource_usage)
        
        neural_score = (
            accuracy_weight * neural_perf.accuracy +
            latency_weight * (1 - neural_perf.latency / max_latency if max_latency > 0 else 1) +
            resource_weight * (1 - neural_perf.resource_usage / max_resource if max_resource > 0 else 1)
        )
        
        algo_score = (
            accuracy_weight * algo_perf.accuracy +
            latency_weight * (1 - algo_perf.latency / max_latency if max_latency > 0 else 1) +
            resource_weight * (1 - algo_perf.resource_usage / max_resource if max_resource > 0 else 1)
        )
        
        # Select best implementation
        best_impl = "neural" if neural_score > algo_score else "algorithmic"
        
        # Log selection
        self.logger.info(f"Selected {best_impl} implementation for {function_name}")
        self.logger.info(f"  Neural score: {neural_score:.4f}")
        self.logger.info(f"  Algorithmic score: {algo_score:.4f}")
        
        # Update current implementation
        self.current_implementation[function_name] = best_impl
        
        return best_impl
    
    def get_implementation_history(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Get performance history for a function.
        
        Args:
            function_name: Name of the function to get history for.
            
        Returns:
            List of performance metric dictionaries.
        """
        if function_name not in self.performance_history:
            return []
        
        return [perf.to_dict() for perf in self.performance_history[function_name]]
    
    def get_current_implementation(self, function_name: str) -> str:
        """
        Get the currently selected implementation for a function.
        
        Args:
            function_name: Name of the function to get implementation for.
            
        Returns:
            Implementation type ("neural" or "algorithmic").
        """
        return self.current_implementation.get(function_name, "algorithmic")
    
    def clear_history(self, function_name: Optional[str] = None) -> None:
        """
        Clear performance history.
        
        Args:
            function_name: Name of the function to clear history for.
                          If None, clears history for all functions.
        """
        if function_name is None:
            self.performance_history = {}
            self.logger.info("Cleared all performance history")
        elif function_name in self.performance_history:
            self.performance_history[function_name] = []
            self.logger.info(f"Cleared performance history for {function_name}")
    
    def get_performance_trends(self, 
                              function_name: str,
                              window_size: int = 10) -> Dict[str, List[float]]:
        """
        Get performance trends over time.
        
        Args:
            function_name: Name of the function to get trends for.
            window_size: Number of recent measurements to include.
            
        Returns:
            Dictionary with lists of performance metrics over time.
        """
        if function_name not in self.performance_history:
            return {
                "neural_accuracy": [],
                "neural_latency": [],
                "algorithmic_accuracy": [],
                "algorithmic_latency": []
            }
        
        # Filter and sort performance history
        history = self.performance_history[function_name]
        history.sort(key=lambda p: p.timestamp)
        
        # Get recent history
        recent_history = history[-window_size*2:]  # *2 because we have two implementations
        
        # Separate neural and algorithmic metrics
        neural_accuracy = []
        neural_latency = []
        algo_accuracy = []
        algo_latency = []
        
        for perf in recent_history:
            if perf.implementation_type == "neural":
                neural_accuracy.append(perf.accuracy)
                neural_latency.append(perf.latency)
            else:
                algo_accuracy.append(perf.accuracy)
                algo_latency.append(perf.latency)
        
        # Trim to window size
        neural_accuracy = neural_accuracy[-window_size:]
        neural_latency = neural_latency[-window_size:]
        algo_accuracy = algo_accuracy[-window_size:]
        algo_latency = algo_latency[-window_size:]
        
        return {
            "neural_accuracy": neural_accuracy,
            "neural_latency": neural_latency,
            "algorithmic_accuracy": algo_accuracy,
            "algorithmic_latency": algo_latency
        }