"""
Implementation Validator: Validates and tests dual implementations.

This module provides tools for validating and testing dual implementations,
ensuring that neural implementations produce correct results compared to
algorithmic implementations.

References:
- Requirements 1.2, 1.3, 1.4, 1.5 from MCP System Upgrade specification
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from .dual_implementation import DualImplementation, ImplementationType

# Type variables for generic implementations
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')


@dataclass
class ValidationResult:
    """Result of validating a dual implementation."""
    component: str
    passed: bool
    accuracy: float  # 0.0-1.0
    error_count: int
    total_tests: int
    execution_time: float
    details: List[Dict[str, Any]] = field(default_factory=list)


class ImplementationValidator(Generic[T_Input, T_Output]):
    """
    Validates and tests dual implementations.
    
    This class provides tools for validating and testing dual implementations,
    ensuring that neural implementations produce correct results compared to
    algorithmic implementations.
    """
    
    def __init__(self, implementation: DualImplementation[T_Input, T_Output], logger=None):
        """
        Initialize the implementation validator.
        
        Args:
            implementation: Dual implementation to validate
            logger: Logger to use, or None to create a new one
        """
        self.implementation = implementation
        self.logger = logger or logging.getLogger(f"ImplementationValidator.{implementation.name}")
    
    async def validate(self, 
                     test_cases: List[Tuple[T_Input, Optional[T_Output]]],
                     equality_check: Callable[[T_Output, T_Output], bool] = None) -> ValidationResult:
        """
        Validate the implementation against test cases.
        
        Args:
            test_cases: List of (input, expected_output) tuples
            equality_check: Function to check if outputs are equal, or None to use ==
            
        Returns:
            Validation result
        """
        if equality_check is None:
            equality_check = lambda a, b: a == b
        
        start_time = time.time()
        error_count = 0
        details = []
        
        # Force algorithmic implementation for expected outputs
        original_impl = self.implementation.current_implementation
        self.implementation.current_implementation = ImplementationType.ALGORITHMIC
        
        # Process test cases
        for i, (input_data, expected_output) in enumerate(test_cases):
            try:
                # Get expected output if not provided
                if expected_output is None:
                    expected_output = await self.implementation._process_algorithmic(input_data)
                
                # Test neural implementation
                self.implementation.current_implementation = ImplementationType.NEURAL
                neural_output = await self.implementation._process_neural(input_data)
                
                # Check if outputs match
                outputs_match = equality_check(neural_output, expected_output)
                
                if not outputs_match:
                    error_count += 1
                    self.logger.warning(
                        f"Test case {i}: Neural output does not match expected output. "
                        f"Expected: {expected_output}, Got: {neural_output}"
                    )
                
                details.append({
                    "test_case": i,
                    "input": input_data,
                    "expected_output": expected_output,
                    "neural_output": neural_output,
                    "passed": outputs_match
                })
                
            except Exception as e:
                error_count += 1
                self.logger.error(f"Test case {i}: Error in neural implementation: {e}")
                
                details.append({
                    "test_case": i,
                    "input": input_data,
                    "expected_output": expected_output,
                    "error": str(e),
                    "passed": False
                })
        
        # Restore original implementation
        self.implementation.current_implementation = original_impl
        
        # Calculate accuracy
        accuracy = 1.0 - (error_count / len(test_cases)) if test_cases else 1.0
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            component=self.implementation.name,
            passed=error_count == 0,
            accuracy=accuracy,
            error_count=error_count,
            total_tests=len(test_cases),
            execution_time=execution_time,
            details=details
        )
    
    async def generate_test_cases(self, 
                                num_cases: int, 
                                input_generator: Callable[[], T_Input]) -> List[Tuple[T_Input, T_Output]]:
        """
        Generate test cases using the algorithmic implementation.
        
        Args:
            num_cases: Number of test cases to generate
            input_generator: Function to generate random inputs
            
        Returns:
            List of (input, expected_output) tuples
        """
        test_cases = []
        
        # Force algorithmic implementation
        original_impl = self.implementation.current_implementation
        self.implementation.current_implementation = ImplementationType.ALGORITHMIC
        
        # Generate test cases
        for _ in range(num_cases):
            input_data = input_generator()
            try:
                output_data = await self.implementation._process_algorithmic(input_data)
                test_cases.append((input_data, output_data))
            except Exception as e:
                self.logger.error(f"Error generating test case: {e}")
        
        # Restore original implementation
        self.implementation.current_implementation = original_impl
        
        return test_cases
    
    async def benchmark(self, 
                      test_cases: List[T_Input], 
                      iterations: int = 1) -> Dict[str, Any]:
        """
        Benchmark both implementations.
        
        Args:
            test_cases: List of inputs to test
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        results = {
            "component": self.implementation.name,
            "iterations": iterations,
            "test_cases": len(test_cases),
            "algorithmic": {
                "total_time": 0.0,
                "avg_time": 0.0,
                "errors": 0
            },
            "neural": {
                "total_time": 0.0,
                "avg_time": 0.0,
                "errors": 0
            }
        }
        
        # Benchmark algorithmic implementation
        self.implementation.current_implementation = ImplementationType.ALGORITHMIC
        algo_start_time = time.time()
        
        for _ in range(iterations):
            for input_data in test_cases:
                try:
                    await self.implementation._process_algorithmic(input_data)
                except Exception:
                    results["algorithmic"]["errors"] += 1
        
        results["algorithmic"]["total_time"] = time.time() - algo_start_time
        results["algorithmic"]["avg_time"] = results["algorithmic"]["total_time"] / (iterations * len(test_cases))
        
        # Benchmark neural implementation
        self.implementation.current_implementation = ImplementationType.NEURAL
        neural_start_time = time.time()
        
        for _ in range(iterations):
            for input_data in test_cases:
                try:
                    await self.implementation._process_neural(input_data)
                except Exception:
                    results["neural"]["errors"] += 1
        
        results["neural"]["total_time"] = time.time() - neural_start_time
        results["neural"]["avg_time"] = results["neural"]["total_time"] / (iterations * len(test_cases))
        
        # Calculate speedup
        if results["algorithmic"]["avg_time"] > 0:
            results["speedup"] = results["algorithmic"]["avg_time"] / results["neural"]["avg_time"]
        else:
            results["speedup"] = 0.0
        
        return results


# Example usage
async def test_implementation_validator():
    """Test the implementation validator."""
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
            # Simulate slightly different results
            return input_data * 2 + (1 if input_data % 10 == 0 else 0)
    
    # Create implementation
    impl = ExampleDualImplementation("example_doubler")
    
    # Create validator
    validator = ImplementationValidator(impl)
    
    # Generate test cases
    test_cases = await validator.generate_test_cases(10, lambda: random.randint(1, 100))
    
    # Validate implementation
    result = await validator.validate(test_cases)
    
    print("\nValidation Result:")
    print(f"Component: {result.component}")
    print(f"Passed: {result.passed}")
    print(f"Accuracy: {result.accuracy:.2f}")
    print(f"Error count: {result.error_count}/{result.total_tests}")
    print(f"Execution time: {result.execution_time:.4f}s")
    
    # Benchmark implementation
    benchmark_inputs = [random.randint(1, 100) for _ in range(10)]
    benchmark_result = await validator.benchmark(benchmark_inputs, iterations=5)
    
    print("\nBenchmark Result:")
    print(f"Algorithmic avg time: {benchmark_result['algorithmic']['avg_time']:.6f}s")
    print(f"Neural avg time: {benchmark_result['neural']['avg_time']:.6f}s")
    print(f"Speedup: {benchmark_result['speedup']:.2f}x")


if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_implementation_validator())