"""
Test script for the neural vs. algorithmic comparison framework.

This script demonstrates how to use the comparison framework to compare
neural network implementations against algorithmic implementations.
"""

import logging
import random
import time
from typing import Any, Dict, List, Tuple

from src.mcp.tests.neural_algorithmic_comparison.comparison_framework import ComparisonFramework

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestComparison")

def generate_test_data(num_samples: int = 100) -> List[Tuple[float, float]]:
    """
    Generate test data for hormone calculations.
    
    Args:
        num_samples: Number of test samples to generate.
        
    Returns:
        List of (input, expected_output) tuples.
    """
    test_data = []
    
    for _ in range(num_samples):
        # Generate random input value between 0 and 1
        input_value = random.random()
        
        # Expected output is a simple function of the input
        # In a real scenario, this would be the ground truth
        expected_output = 0.5 * input_value + 0.2 * (input_value ** 2)
        
        test_data.append((input_value, expected_output))
    
    return test_data

def algorithmic_implementation(input_value: float) -> float:
    """
    Algorithmic implementation of a hormone calculation.
    
    Args:
        input_value: Input value for the calculation.
        
    Returns:
        Calculated hormone level.
    """
    # Simple algorithmic implementation
    # In a real scenario, this would be more complex
    return 0.5 * input_value + 0.2 * (input_value ** 2)

def neural_implementation(input_value: float) -> float:
    """
    Neural network implementation of a hormone calculation.
    
    Args:
        input_value: Input value for the calculation.
        
    Returns:
        Calculated hormone level.
    """
    # Simulate a neural network with some error
    # In a real scenario, this would use an actual neural network
    result = 0.5 * input_value + 0.2 * (input_value ** 2)
    
    # Add some noise to simulate neural network error
    error = random.gauss(0, 0.01)
    result += error
    
    # Simulate neural network latency
    time.sleep(0.001)
    
    return max(0.0, min(1.0, result))  # Clamp to [0, 1]

def test_comparison_framework():
    """Test the comparison framework with simulated implementations."""
    logger.info("Testing neural vs. algorithmic comparison framework")
    
    # Create comparison framework
    framework = ComparisonFramework()
    
    # Generate test data
    test_data = generate_test_data(100)
    
    # Define functions to test
    functions = [
        {
            "name": "dopamine_production",
            "neural": neural_implementation,
            "algorithmic": algorithmic_implementation
        },
        {
            "name": "serotonin_production",
            "neural": neural_implementation,
            "algorithmic": algorithmic_implementation
        }
    ]
    
    # Test each function
    for func in functions:
        logger.info(f"Testing {func['name']}...")
        
        # Measure performance
        performance = framework.measure_performance(
            function_name=func["name"],
            neural_impl=func["neural"],
            algo_impl=func["algorithmic"],
            test_data=test_data,
            context={"test_run": True}
        )
        
        # Select best implementation
        best_impl = framework.select_best_implementation(func["name"])
        logger.info(f"Best implementation for {func['name']}: {best_impl}")
        
        # Get performance trends
        trends = framework.get_performance_trends(func["name"])
        logger.info(f"Performance trends for {func['name']}: {trends}")
    
    logger.info("Comparison framework test completed successfully")

if __name__ == "__main__":
    test_comparison_framework()