#!/usr/bin/env python3
"""
Simple test script for neural network models.
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

def test_hormone_integration():
    """Test hormone neural integration."""
    try:
        from mcp.neural_network_models.hormone_neural_integration import HormoneNeuralIntegration, HormoneConfig
        
        logger.info("Creating hormone integration model...")
        config = HormoneConfig(input_dim=10, hidden_dim=32, output_dim=4)
        model = HormoneNeuralIntegration(config)
        
        # Test updating hormone levels
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
        
        logger.info("Updating hormone levels...")
        hormone_levels = model.update_hormone_levels(metrics)
        
        logger.info(f"Hormone levels: {hormone_levels}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing hormone integration: {e}")
        return False

def test_diffusion_model():
    """Test diffusion model."""
    try:
        import torch
        from mcp.neural_network_models.diffusion_model import DiffusionModel, DiffusionConfig
        
        logger.info("Creating diffusion model...")
        config = DiffusionConfig(input_dim=8, hidden_dim=32, timesteps=10, epochs=1)
        model = DiffusionModel(config)
        
        # Create sample embeddings
        embeddings = torch.randn(10, 8)
        
        logger.info("Training diffusion model...")
        model.train(embeddings, epochs=1)
        
        logger.info("Generating samples...")
        samples = model.sample(2)
        
        logger.info(f"Generated samples shape: {samples.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing diffusion model: {e}")
        return False

def test_brain_state_integration():
    """Test brain state integration."""
    try:
        import torch
        from mcp.neural_network_models.brain_state_integration import BrainStateIntegration, BrainStateConfig
        
        logger.info("Creating brain state integration...")
        config = BrainStateConfig(embedding_dim=8, state_dim=16)
        model = BrainStateIntegration(config)
        
        # Create sample embedding and metrics
        embedding = torch.randn(8)
        metrics = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'error_count': 5
        }
        
        logger.info("Updating brain state...")
        state = model.update_brain_state(embedding, metrics)
        
        logger.info(f"Brain state shape: {state.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing brain state integration: {e}")
        return False

if __name__ == "__main__":
    print("Testing neural network models...")
    
    # Test hormone integration
    print("\n=== Testing Hormone Integration ===")
    hormone_success = test_hormone_integration()
    print(f"Hormone Integration Test: {'Success' if hormone_success else 'Failed'}")
    
    # Test diffusion model
    print("\n=== Testing Diffusion Model ===")
    diffusion_success = test_diffusion_model()
    print(f"Diffusion Model Test: {'Success' if diffusion_success else 'Failed'}")
    
    # Test brain state integration
    print("\n=== Testing Brain State Integration ===")
    brain_state_success = test_brain_state_integration()
    print(f"Brain State Integration Test: {'Success' if brain_state_success else 'Failed'}")
    
    # Overall result
    overall_success = hormone_success and diffusion_success and brain_state_success
    print(f"\nOverall Test Result: {'Success' if overall_success else 'Failed'}")
    
    sys.exit(0 if overall_success else 1)