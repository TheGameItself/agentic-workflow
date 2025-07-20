"""
Fallback Manager: Manages graceful degradation when neural implementations fail.

This module provides a centralized system for managing fallbacks when neural
implementations fail, with circuit breaker patterns and exponential backoff.

References:
- Requirements 1.5, 1.9 from MCP System Upgrade specification
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .dual_implementation import DualImplementation, DualImplementationRegistry, ImplementationType


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, not allowing neural implementation
    HALF_OPEN = "half_open"  # Testing if neural implementation works again


@dataclass
class CircuitBreaker:
    """Circuit breaker for neural implementation fallbacks."""
    component: str
    failure_threshold: int = 3  # Number of failures before opening circuit
    reset_timeout: float = 60.0  # Seconds before trying neural implementation again
    exponential_backoff: bool = True  # Whether to use exponential backoff
    max_backoff: float = 3600.0  # Maximum backoff time in seconds
    
    # State
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    current_backoff: float = 0.0
    
    # History
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    state_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_failure(self, error: Exception) -> None:
        """
        Record a failure and potentially open the circuit.
        
        Args:
            error: Exception that caused the failure
        """
        current_time = time.time()
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Record failure
        self.failure_history.append({
            "timestamp": current_time,
            "error": str(error),
            "state": self.state.value
        })
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self._change_state(CircuitState.OPEN)
            
            # Calculate backoff time
            if self.exponential_backoff:
                self.current_backoff = min(
                    self.reset_timeout * (2 ** (self.failure_count - self.failure_threshold)),
                    self.max_backoff
                )
            else:
                self.current_backoff = self.reset_timeout
    
    def record_success(self) -> None:
        """Record a successful operation and potentially close the circuit."""
        current_time = time.time()
        self.last_success_time = current_time
        
        # If in half-open state, close the circuit
        if self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.CLOSED)
            self.failure_count = 0
            self.current_backoff = 0.0
    
    def can_use_neural(self) -> bool:
        """
        Check if the neural implementation can be used.
        
        Returns:
            True if the neural implementation can be used, False otherwise
        """
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if we should try half-open state
            if current_time - self.last_failure_time > self.current_backoff:
                self._change_state(CircuitState.HALF_OPEN)
                return True
            else:
                return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _change_state(self, new_state: CircuitState) -> None:
        """
        Change the circuit state.
        
        Args:
            new_state: New circuit state
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            
            # Record state change
            self.state_changes.append({
                "timestamp": time.time(),
                "old_state": old_state.value,
                "new_state": new_state.value,
                "failure_count": self.failure_count,
                "backoff": self.current_backoff
            })
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the circuit breaker.
        
        Returns:
            Dictionary with circuit breaker status
        """
        current_time = time.time()
        time_since_failure = current_time - self.last_failure_time if self.last_failure_time > 0 else None
        time_since_success = current_time - self.last_success_time if self.last_success_time > 0 else None
        
        return {
            "component": self.component,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_since_failure": time_since_failure,
            "time_since_success": time_since_success,
            "current_backoff": self.current_backoff,
            "can_use_neural": self.can_use_neural(),
            "failure_threshold": self.failure_threshold,
            "reset_timeout": self.reset_timeout,
            "exponential_backoff": self.exponential_backoff,
            "max_backoff": self.max_backoff
        }


class FallbackManager:
    """
    Manages fallbacks when neural implementations fail.
    
    This class provides a centralized system for managing fallbacks when neural
    implementations fail, with circuit breaker patterns and exponential backoff.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FallbackManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the fallback manager."""
        if self._initialized:
            return
            
        self.logger = logging.getLogger("FallbackManager")
        self.registry = DualImplementationRegistry()
        
        # Circuit breakers for each component
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Default circuit breaker configuration
        self.default_failure_threshold = 3
        self.default_reset_timeout = 60.0
        self.default_exponential_backoff = True
        self.default_max_backoff = 3600.0
        
        # Initialize circuit breakers for all registered implementations
        self._initialize_circuit_breakers()
        
        self.logger.info("Fallback manager initialized")
        self._initialized = True
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for all registered implementations."""
        for name in self.registry.get_all().keys():
            self._get_or_create_circuit_breaker(name)
    
    def _get_or_create_circuit_breaker(self, component: str) -> CircuitBreaker:
        """
        Get or create a circuit breaker for a component.
        
        Args:
            component: Component name
            
        Returns:
            Circuit breaker for the component
        """
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(
                component=component,
                failure_threshold=self.default_failure_threshold,
                reset_timeout=self.default_reset_timeout,
                exponential_backoff=self.default_exponential_backoff,
                max_backoff=self.default_max_backoff
            )
        
        return self.circuit_breakers[component]
    
    def record_failure(self, component: str, error: Exception) -> None:
        """
        Record a failure for a component.
        
        Args:
            component: Component name
            error: Exception that caused the failure
        """
        circuit_breaker = self._get_or_create_circuit_breaker(component)
        circuit_breaker.record_failure(error)
        
        self.logger.warning(
            f"Neural implementation failure for {component}: {error}. "
            f"Circuit breaker state: {circuit_breaker.state.value}, "
            f"Failure count: {circuit_breaker.failure_count}"
        )
    
    def record_success(self, component: str) -> None:
        """
        Record a success for a component.
        
        Args:
            component: Component name
        """
        circuit_breaker = self._get_or_create_circuit_breaker(component)
        
        # Only log if state changes
        if circuit_breaker.state != CircuitState.CLOSED:
            old_state = circuit_breaker.state
            circuit_breaker.record_success()
            
            if circuit_breaker.state != old_state:
                self.logger.info(
                    f"Neural implementation success for {component}. "
                    f"Circuit breaker state changed from {old_state.value} to {circuit_breaker.state.value}"
                )
        else:
            circuit_breaker.record_success()
    
    def can_use_neural(self, component: str) -> bool:
        """
        Check if the neural implementation can be used for a component.
        
        Args:
            component: Component name
            
        Returns:
            True if the neural implementation can be used, False otherwise
        """
        circuit_breaker = self._get_or_create_circuit_breaker(component)
        return circuit_breaker.can_use_neural()
    
    def get_circuit_breaker_status(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a circuit breaker.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with circuit breaker status, or None if not found
        """
        if component in self.circuit_breakers:
            return self.circuit_breakers[component].get_status()
        else:
            return None
    
    def get_all_circuit_breaker_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all circuit breakers.
        
        Returns:
            Dictionary mapping component names to circuit breaker statuses
        """
        return {name: breaker.get_status() for name, breaker in self.circuit_breakers.items()}
    
    def configure_circuit_breaker(self, 
                                component: str, 
                                failure_threshold: Optional[int] = None,
                                reset_timeout: Optional[float] = None,
                                exponential_backoff: Optional[bool] = None,
                                max_backoff: Optional[float] = None) -> None:
        """
        Configure a circuit breaker.
        
        Args:
            component: Component name
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before trying neural implementation again
            exponential_backoff: Whether to use exponential backoff
            max_backoff: Maximum backoff time in seconds
        """
        circuit_breaker = self._get_or_create_circuit_breaker(component)
        
        if failure_threshold is not None:
            circuit_breaker.failure_threshold = failure_threshold
        
        if reset_timeout is not None:
            circuit_breaker.reset_timeout = reset_timeout
        
        if exponential_backoff is not None:
            circuit_breaker.exponential_backoff = exponential_backoff
        
        if max_backoff is not None:
            circuit_breaker.max_backoff = max_backoff
        
        self.logger.info(f"Configured circuit breaker for {component}")
    
    def reset_circuit_breaker(self, component: str) -> None:
        """
        Reset a circuit breaker to closed state.
        
        Args:
            component: Component name
        """
        if component in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[component]
            old_state = circuit_breaker.state
            
            circuit_breaker.state = CircuitState.CLOSED
            circuit_breaker.failure_count = 0
            circuit_breaker.current_backoff = 0.0
            
            circuit_breaker.state_changes.append({
                "timestamp": time.time(),
                "old_state": old_state.value,
                "new_state": CircuitState.CLOSED.value,
                "failure_count": 0,
                "backoff": 0.0,
                "reason": "manual_reset"
            })
            
            self.logger.info(
                f"Manually reset circuit breaker for {component} from {old_state.value} to CLOSED"
            )
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        for component in self.circuit_breakers.keys():
            self.reset_circuit_breaker(component)
    
    def get_failure_history(self, component: str) -> List[Dict[str, Any]]:
        """
        Get the failure history for a component.
        
        Args:
            component: Component name
            
        Returns:
            List of failure events, or empty list if not found
        """
        if component in self.circuit_breakers:
            return self.circuit_breakers[component].failure_history.copy()
        else:
            return []
    
    def get_state_change_history(self, component: str) -> List[Dict[str, Any]]:
        """
        Get the state change history for a component.
        
        Args:
            component: Component name
            
        Returns:
            List of state change events, or empty list if not found
        """
        if component in self.circuit_breakers:
            return self.circuit_breakers[component].state_changes.copy()
        else:
            return []
    
    def get_open_circuits(self) -> List[str]:
        """
        Get a list of components with open circuits.
        
        Returns:
            List of component names with open circuits
        """
        return [name for name, breaker in self.circuit_breakers.items() 
                if breaker.state == CircuitState.OPEN]
    
    def get_half_open_circuits(self) -> List[str]:
        """
        Get a list of components with half-open circuits.
        
        Returns:
            List of component names with half-open circuits
        """
        return [name for name, breaker in self.circuit_breakers.items() 
                if breaker.state == CircuitState.HALF_OPEN]
    
    async def monitor_loop(self, interval: float = 60.0) -> None:
        """
        Run a monitoring loop that periodically checks circuit breakers.
        
        Args:
            interval: Check interval in seconds
        """
        self.logger.info(f"Starting circuit breaker monitoring loop with interval {interval}s")
        
        while True:
            try:
                # Check for circuit breakers that can be half-opened
                current_time = time.time()
                
                for component, breaker in self.circuit_breakers.items():
                    if breaker.state == CircuitState.OPEN:
                        if current_time - breaker.last_failure_time > breaker.current_backoff:
                            self.logger.info(
                                f"Circuit breaker for {component} can be half-opened "
                                f"(backoff expired: {breaker.current_backoff:.1f}s)"
                            )
                
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in circuit breaker monitoring loop: {e}")
                await asyncio.sleep(interval)


# Example usage
async def test_fallback_manager():
    """Test the fallback manager."""
    from .dual_implementation import DualImplementation
    
    class ExampleDualImplementation(DualImplementation[int, int]):
        """Example dual implementation for testing."""
        
        async def _process_algorithmic(self, input_data: int) -> int:
            """Algorithmic implementation: double the input."""
            await asyncio.sleep(0.001)  # Simulate processing time
            return input_data * 2
        
        async def _process_neural(self, input_data: int) -> int:
            """Neural implementation: double the input with small variation."""
            await asyncio.sleep(0.002)  # Simulate longer processing time
            # Simulate failures for certain inputs
            if input_data % 10 == 0:
                raise ValueError(f"Simulated failure for input {input_data}")
            return input_data * 2 + (1 if input_data % 5 == 0 else 0)
    
    # Create and register implementation
    impl = ExampleDualImplementation("example_doubler")
    registry = DualImplementationRegistry()
    registry.register(impl)
    
    # Create fallback manager
    manager = FallbackManager()
    
    # Configure circuit breaker
    manager.configure_circuit_breaker(
        component="example_doubler",
        failure_threshold=2,
        reset_timeout=5.0,
        exponential_backoff=True,
        max_backoff=30.0
    )
    
    # Process some data
    for i in range(30):
        try:
            # Check if we can use neural implementation
            if manager.can_use_neural("example_doubler"):
                try:
                    result = await impl._process_neural(i)
                    print(f"Neural - Input: {i}, Result: {result}")
                    manager.record_success("example_doubler")
                except Exception as e:
                    print(f"Neural failed - Input: {i}, Error: {e}")
                    manager.record_failure("example_doubler", e)
                    result = await impl._process_algorithmic(i)
                    print(f"Fallback to algorithmic - Input: {i}, Result: {result}")
            else:
                result = await impl._process_algorithmic(i)
                print(f"Using algorithmic (circuit open) - Input: {i}, Result: {result}")
            
            # Print circuit breaker status every 5 iterations
            if i % 5 == 0:
                status = manager.get_circuit_breaker_status("example_doubler")
                print(f"Circuit breaker status: {status['state']}, "
                      f"Failures: {status['failure_count']}, "
                      f"Backoff: {status['current_backoff']:.1f}s")
            
            # Add delay to see circuit breaker in action
            await asyncio.sleep(1.0)
            
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    # Print final status
    status = manager.get_circuit_breaker_status("example_doubler")
    print("\nFinal circuit breaker status:")
    for key, value in status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_fallback_manager())