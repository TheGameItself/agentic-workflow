#!/usr/bin/env python3
"""
Test script for hormone neural integration.
"""

import os
import sys
from pathlib import Path

# Add core/src to path
core_src = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_src))

# Create data directories
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test")

def main():
    """Test hormone neural integration."""
    try:
        print("Importing hormone_neural_integration...")
        from mcp.neural_network_models.hormone_neural_integration import HormoneNeuralIntegration, HormoneConfig
        
        print("Creating hormone integration model...")
        config = HormoneConfig(input_dim=10, hidden_dim=32, output_dim=4)
        model = HormoneNeuralIntegration(config)
        
        # Test updating hormone levels with rule-based approach
        metrics = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'error_count': 5,
            'response_time': 0.5,
            'uptime': 3600.0,
            'throughput': 100.0,
            'active_lobes': 3,
            'total_requests': 1000,
            'queue_size': 5
        }
        
        print("Updating hormone levels...")
        hormone_levels = model.update_hormone_levels(metrics)
        
        print(f"Hormone levels: {hormone_levels}")
        
        # Generate training data
        print("Generating training data...")
        metrics_history = []
        for i in range(100):
            # Create variations of metrics
            metric_variation = {k: v * (0.8 + 0.4 * (i / 100)) for k, v in metrics.items()}
            metrics_history.append(metric_variation)
        
        inputs, outputs = model.generate_training_data(metrics_history)
        print(f"Generated training data: inputs shape {inputs.shape}, outputs shape {outputs.shape}")
        
        # Train a simple model (with minimal epochs for testing)
        print("Training hormone neural network...")
        model.train(inputs, outputs, epochs=1)
        
        # Test the trained model
        print("Testing trained model...")
        new_metrics = {
            'cpu_usage': 75.0,
            'memory_usage': 80.0,
            'error_count': 10,
            'response_time': 1.0,
            'uptime': 7200.0,
            'throughput': 80.0,
            'active_lobes': 2,
            'total_requests': 2000,
            'queue_size': 10
        }
        
        new_hormone_levels = model.update_hormone_levels(new_metrics)
        print(f"New hormone levels: {new_hormone_levels}")
        
        # Test feedback mechanism
        print("Testing feedback mechanism...")
        model.provide_feedback('stress', -0.5)  # Reduce stress
        model.provide_feedback('efficiency', 0.5)  # Increase efficiency
        
        adjusted_hormone_levels = model.update_hormone_levels(new_metrics)
        print(f"Adjusted hormone levels: {adjusted_hormone_levels}")
        
        print("Hormone neural integration test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())