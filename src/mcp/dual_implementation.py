"""
Dual Implementation Framework: Base classes for neural/algorithmic dual implementations.

This module provides the core infrastructure for implementing both algorithmic and neural
network solutions for critical functions, with automatic performance comparison and switching.

References:
- Requirements 1.1, 1.4, 5.1, 5.2 from MCP System Upgrade specification
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

# Type variables for generic implementations
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')
T_Model = TypeVar('T_Model')


class ImplementationType(Enum):
    """Types of implementations available in the dual implementation framework."""
    ALGORITHMIC = "algorithmic"
    NEURAL = "neural"


@dataclass
class PerformanceMetrics:
    """Performance metrics for comparing implementations."""
    accuracy: float = 0.0  # Correctness of results (0.0-1.0)
    speed: float = 0.0  # Operations per second
    resource_usage: float = 0.0  # CPU/memory utilization (0.0-1.0)
    error_rate: float = 0.0  # Rate of errors or exceptions (0.0-1.0)
    confidence_score: float = 0.0  # Confidence in results (0.0-1.0)
    timestamp: float = field(default_factory=time.time)  # When metrics were collected
    
    def weighted_score(self, weights: Dict[str, float] = None) -> float:
        """
        Calculate a weighted score based on the metrics.
        
        Args:
            weights: Dictionary mapping metric names to weights (default: equal weights)
            
        Returns:
            Weighted score (higher is better)
        """
        if weights is None:
            weights = {
                "accuracy": 0.4,
                "speed": 0.2,
                "resource_usage": 0.2,  # Lower is better, inverted in calculation
                "error_rate": 0.1,  # Lower is better, inverted in calculation
                "confidence_score": 0.1
            }
            
        score = (
            weights.get("accuracy", 0.0) * self.accuracy +
            weights.get("speed", 0.0) * min(1.0, self.speed / 1000.0) +  # Normalize speed
            weights.get("resource_usage", 0.0) * (1.0 - self.resource_usage) +  # Invert (lower is better)
            weights.get("error_rate", 0.0) * (1.0 - self.error_rate) +  # Invert (lower is better)
            weights.get("confidence_score", 0.0) * self.confidence_score
        )
        
        return score


@dataclass
class ComparisonResult:
    """Result of comparing algorithmic and neural implementations."""
    algorithmic_metrics: PerformanceMetrics
    neural_metrics: PerformanceMetrics
    better_implementation: ImplementationType
    improvement_factor: float  # How much better the winning implementation is
    comparison_context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    @property
    def is_significant_improvement(self) -> bool:
        """
        Determine if the improvement is significant enough to warrant switching.
        
        Returns:
            True if the improvement is significant, False otherwise
        """
        # Require at least 10% improvement to switch
        return self.improvement_factor >= 1.1


class DualImplementation(Generic[T_Input, T_Output], ABC):
    """
    Base class for dual implementation (algorithmic and neural) of a function.
    
    This abstract class defines the interface for implementing both algorithmic
    and neural network solutions for a function, with automatic performance
    comparison and switching.
    """
    
    def __init__(self, 
                 name: str,
                 initial_implementation: ImplementationType = ImplementationType.ALGORITHMIC,
                 performance_threshold: float = 1.1,
                 comparison_frequency: int = 100,
                 weights: Dict[str, float] = None):
        """
        Initialize the dual implementation.
        
        Args:
            name: Name of the implementation (for logging and tracking)
            initial_implementation: Which implementation to use initially
            performance_threshold: How much better the neural implementation must be to switch
            comparison_frequency: How often to compare implementations (in calls)
            weights: Weights for different performance metrics
        """
        self.name = name
        self.current_implementation = initial_implementation
        self.performance_threshold = performance_threshold
        self.comparison_frequency = comparison_frequency
        self.weights = weights or {
            "accuracy": 0.4,
            "speed": 0.2,
            "resource_usage": 0.2,
            "error_rate": 0.1,
            "confidence_score": 0.1
        }
        
        # Performance tracking
        self.call_count = 0
        self.algorithmic_metrics: List[PerformanceMetrics] = []
        self.neural_metrics: List[PerformanceMetrics] = []
        self.comparison_results: List[ComparisonResult] = []
        
        # Fallback tracking
        self.fallback_count = 0
        self.last_fallback_time = 0.0
        self.in_fallback_mode = False
        self.fallback_cooldown = 60.0  # Seconds to wait before trying neural again after fallback
        
        # Logging
        self.logger = logging.getLogger(f"DualImplementation.{name}")
        self.logger.info(f"Initialized with {initial_implementation.value} implementation")
    
    async def process(self, input_data: T_Input) -> T_Output:
        """
        Process input data using the current implementation.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        self.call_count += 1
        
        # Check if we should compare implementations
        should_compare = (
            self.call_count % self.comparison_frequency == 0 and
            not self.in_fallback_mode
        )
        
        # If in fallback mode, check if cooldown has expired
        if self.in_fallback_mode:
            if time.time() - self.last_fallback_time > self.fallback_cooldown:
                self.logger.info(f"Fallback cooldown expired, attempting to use neural implementation again")
                self.in_fallback_mode = False
                should_compare = True
        
        if should_compare:
            return await self._compare_and_process(input_data)
        else:
            return await self._process_with_current(input_data)
    
    async def _process_with_current(self, input_data: T_Input) -> T_Output:
        """
        Process input data using the current implementation.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        try:
            if self.current_implementation == ImplementationType.ALGORITHMIC:
                start_time = time.time()
                result = await self._process_algorithmic(input_data)
                elapsed = time.time() - start_time
                
                # Record metrics (simplified for algorithmic implementation)
                self.algorithmic_metrics.append(PerformanceMetrics(
                    accuracy=1.0,  # Assume algorithmic is correct
                    speed=1.0 / elapsed if elapsed > 0 else 1000.0,
                    resource_usage=0.1,  # Assume low resource usage
                    error_rate=0.0,
                    confidence_score=1.0,
                    timestamp=time.time()
                ))
                
                return result
            else:
                start_time = time.time()
                result = await self._process_neural(input_data)
                elapsed = time.time() - start_time
                
                # Record metrics (simplified for neural implementation)
                self.neural_metrics.append(PerformanceMetrics(
                    accuracy=0.95,  # Assume slightly less accurate
                    speed=1.0 / elapsed if elapsed > 0 else 1000.0,
                    resource_usage=0.3,  # Assume higher resource usage
                    error_rate=0.01,
                    confidence_score=0.9,
                    timestamp=time.time()
                ))
                
                return result
        except Exception as e:
            # If neural implementation fails, fall back to algorithmic
            if self.current_implementation == ImplementationType.NEURAL:
                self.logger.warning(f"Neural implementation failed: {e}, falling back to algorithmic")
                self.fallback_count += 1
                self.last_fallback_time = time.time()
                self.in_fallback_mode = True
                self.current_implementation = ImplementationType.ALGORITHMIC
                
                # Try again with algorithmic implementation
                return await self._process_algorithmic(input_data)
            else:
                # If algorithmic implementation fails, we have no fallback
                self.logger.error(f"Algorithmic implementation failed: {e}")
                raise
    
    async def _compare_and_process(self, input_data: T_Input) -> T_Output:
        """
        Compare both implementations and process with the better one.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data from the better implementation
        """
        self.logger.info(f"Comparing implementations for {self.name}")
        
        # Process with both implementations
        try:
            # Algorithmic implementation
            start_time = time.time()
            algorithmic_result = await self._process_algorithmic(input_data)
            algorithmic_elapsed = time.time() - start_time
            
            # Neural implementation
            start_time = time.time()
            neural_result = await self._process_neural(input_data)
            neural_elapsed = time.time() - start_time
            
            # Collect metrics
            algorithmic_metrics = await self._collect_algorithmic_metrics(
                input_data, algorithmic_result, algorithmic_elapsed
            )
            
            neural_metrics = await self._collect_neural_metrics(
                input_data, neural_result, neural_elapsed
            )
            
            # Compare implementations
            comparison = self._compare_metrics(algorithmic_metrics, neural_metrics)
            self.comparison_results.append(comparison)
            
            # Decide which implementation to use
            if comparison.better_implementation != self.current_implementation:
                if comparison.is_significant_improvement:
                    self.logger.info(
                        f"Switching from {self.current_implementation.value} to "
                        f"{comparison.better_implementation.value} "
                        f"(improvement factor: {comparison.improvement_factor:.2f})"
                    )
                    self.current_implementation = comparison.better_implementation
                else:
                    self.logger.info(
                        f"Improvement not significant enough to switch "
                        f"({comparison.improvement_factor:.2f} < {self.performance_threshold})"
                    )
            
            # Return result from the current implementation
            if self.current_implementation == ImplementationType.ALGORITHMIC:
                return algorithmic_result
            else:
                return neural_result
            
        except Exception as e:
            self.logger.error(f"Error during comparison: {e}")
            # Fall back to algorithmic implementation
            self.current_implementation = ImplementationType.ALGORITHMIC
            self.in_fallback_mode = True
            self.last_fallback_time = time.time()
            self.fallback_count += 1
            return await self._process_algorithmic(input_data)
    
    def _compare_metrics(self, 
                        algorithmic_metrics: PerformanceMetrics, 
                        neural_metrics: PerformanceMetrics) -> ComparisonResult:
        """
        Compare metrics from both implementations.
        
        Args:
            algorithmic_metrics: Metrics from algorithmic implementation
            neural_metrics: Metrics from neural implementation
            
        Returns:
            Comparison result
        """
        algorithmic_score = algorithmic_metrics.weighted_score(self.weights)
        neural_score = neural_metrics.weighted_score(self.weights)
        
        if neural_score > algorithmic_score:
            better_impl = ImplementationType.NEURAL
            improvement_factor = neural_score / algorithmic_score if algorithmic_score > 0 else 1.5
        else:
            better_impl = ImplementationType.ALGORITHMIC
            improvement_factor = algorithmic_score / neural_score if neural_score > 0 else 1.5
        
        return ComparisonResult(
            algorithmic_metrics=algorithmic_metrics,
            neural_metrics=neural_metrics,
            better_implementation=better_impl,
            improvement_factor=improvement_factor,
            comparison_context={
                "algorithmic_score": algorithmic_score,
                "neural_score": neural_score,
                "weights": self.weights,
                "call_count": self.call_count
            }
        )
    
    async def _collect_algorithmic_metrics(self, 
                                         input_data: T_Input, 
                                         result: T_Output, 
                                         elapsed: float) -> PerformanceMetrics:
        """
        Collect performance metrics for algorithmic implementation.
        
        Args:
            input_data: Input data
            result: Output data
            elapsed: Elapsed time in seconds
            
        Returns:
            Performance metrics
        """
        # Default implementation - override for more accurate metrics
        return PerformanceMetrics(
            accuracy=1.0,  # Assume algorithmic is correct
            speed=1.0 / elapsed if elapsed > 0 else 1000.0,
            resource_usage=0.1,  # Assume low resource usage
            error_rate=0.0,
            confidence_score=1.0,
            timestamp=time.time()
        )
    
    async def _collect_neural_metrics(self, 
                                    input_data: T_Input, 
                                    result: T_Output, 
                                    elapsed: float) -> PerformanceMetrics:
        """
        Collect performance metrics for neural implementation.
        
        Args:
            input_data: Input data
            result: Output data
            elapsed: Elapsed time in seconds
            
        Returns:
            Performance metrics
        """
        # Default implementation - override for more accurate metrics
        return PerformanceMetrics(
            accuracy=0.95,  # Assume slightly less accurate
            speed=1.0 / elapsed if elapsed > 0 else 1000.0,
            resource_usage=0.3,  # Assume higher resource usage
            error_rate=0.01,
            confidence_score=0.9,
            timestamp=time.time()
        )
    
    @abstractmethod
    async def _process_algorithmic(self, input_data: T_Input) -> T_Output:
        """
        Process input data using the algorithmic implementation.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        pass
    
    @abstractmethod
    async def _process_neural(self, input_data: T_Input) -> T_Output:
        """
        Process input data using the neural network implementation.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        # Calculate average metrics
        avg_algorithmic = PerformanceMetrics()
        if self.algorithmic_metrics:
            avg_algorithmic.accuracy = sum(m.accuracy for m in self.algorithmic_metrics) / len(self.algorithmic_metrics)
            avg_algorithmic.speed = sum(m.speed for m in self.algorithmic_metrics) / len(self.algorithmic_metrics)
            avg_algorithmic.resource_usage = sum(m.resource_usage for m in self.algorithmic_metrics) / len(self.algorithmic_metrics)
            avg_algorithmic.error_rate = sum(m.error_rate for m in self.algorithmic_metrics) / len(self.algorithmic_metrics)
            avg_algorithmic.confidence_score = sum(m.confidence_score for m in self.algorithmic_metrics) / len(self.algorithmic_metrics)
        
        avg_neural = PerformanceMetrics()
        if self.neural_metrics:
            avg_neural.accuracy = sum(m.accuracy for m in self.neural_metrics) / len(self.neural_metrics)
            avg_neural.speed = sum(m.speed for m in self.neural_metrics) / len(self.neural_metrics)
            avg_neural.resource_usage = sum(m.resource_usage for m in self.neural_metrics) / len(self.neural_metrics)
            avg_neural.error_rate = sum(m.error_rate for m in self.neural_metrics) / len(self.neural_metrics)
            avg_neural.confidence_score = sum(m.confidence_score for m in self.neural_metrics) / len(self.neural_metrics)
        
        # Calculate switch counts
        switches = 0
        if len(self.comparison_results) > 1:
            for i in range(1, len(self.comparison_results)):
                if self.comparison_results[i].better_implementation != self.comparison_results[i-1].better_implementation:
                    switches += 1
        
        return {
            "name": self.name,
            "current_implementation": self.current_implementation.value,
            "call_count": self.call_count,
            "comparison_count": len(self.comparison_results),
            "switch_count": switches,
            "fallback_count": self.fallback_count,
            "in_fallback_mode": self.in_fallback_mode,
            "algorithmic_metrics": {
                "count": len(self.algorithmic_metrics),
                "accuracy": avg_algorithmic.accuracy,
                "speed": avg_algorithmic.speed,
                "resource_usage": avg_algorithmic.resource_usage,
                "error_rate": avg_algorithmic.error_rate,
                "confidence_score": avg_algorithmic.confidence_score
            },
            "neural_metrics": {
                "count": len(self.neural_metrics),
                "accuracy": avg_neural.accuracy,
                "speed": avg_neural.speed,
                "resource_usage": avg_neural.resource_usage,
                "error_rate": avg_neural.error_rate,
                "confidence_score": avg_neural.confidence_score
            }
        }


class DualImplementationRegistry:
    """Registry for tracking all dual implementations in the system."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DualImplementationRegistry, cls).__new__(cls)
            cls._instance.implementations = {}
            cls._instance.logger = logging.getLogger("DualImplementationRegistry")
        return cls._instance
    
    def register(self, implementation: DualImplementation) -> None:
        """
        Register a dual implementation.
        
        Args:
            implementation: Dual implementation to register
        """
        self.implementations[implementation.name] = implementation
        self.logger.info(f"Registered implementation: {implementation.name}")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a dual implementation.
        
        Args:
            name: Name of the implementation to unregister
        """
        if name in self.implementations:
            del self.implementations[name]
            self.logger.info(f"Unregistered implementation: {name}")
    
    def get(self, name: str) -> Optional[DualImplementation]:
        """
        Get a dual implementation by name.
        
        Args:
            name: Name of the implementation to get
            
        Returns:
            Dual implementation, or None if not found
        """
        return self.implementations.get(name)
    
    def get_all(self) -> Dict[str, DualImplementation]:
        """
        Get all registered dual implementations.
        
        Returns:
            Dictionary mapping names to dual implementations
        """
        return self.implementations.copy()
    
    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance summaries for all implementations.
        
        Returns:
            Dictionary mapping implementation names to performance summaries
        """
        return {name: impl.get_performance_summary() 
                for name, impl in self.implementations.items()}


# Example usage
async def test_dual_implementation():
    """Test the dual implementation framework."""
    
    class ExampleDualImplementation(DualImplementation[int, int]):
        """Example dual implementation for testing."""
        
        async def _process_algorithmic(self, input_data: int) -> int:
            """Algorithmic implementation: double the input."""
            await asyncio.sleep(0.001)  # Simulate processing time
            return input_data * 2
        
        async def _process_neural(self, input_data: int) -> int:
            """Neural implementation: double the input with small variation."""
            await asyncio.sleep(0.002)  # Simulate longer processing time
            # Simulate slightly different results
            return input_data * 2 + (1 if input_data % 10 == 0 else 0)
    
    # Create and register implementation
    impl = ExampleDualImplementation("example_doubler")
    registry = DualImplementationRegistry()
    registry.register(impl)
    
    # Process some data
    for i in range(200):
        result = await impl.process(i)
        if i % 50 == 0:
            print(f"Input: {i}, Result: {result}")
    
    # Print performance summary
    summary = impl.get_performance_summary()
    print("\nPerformance Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_dual_implementation())