#!/usr/bin/env python3
"""
Simple test script to verify neural network models can be imported.
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

print("Testing neural network model imports...")

try:
    print("Importing neural_network_models package...")
    from mcp.neural_network_models import check_dependencies
    
    print("Checking dependencies...")
    dependencies = check_dependencies()
    print(f"Dependencies: {dependencies}")
    
    print("Importing hormone_neural_integration...")
    from mcp.neural_network_models.hormone_neural_integration import HormoneNeuralIntegration
    
    print("Importing diffusion_model...")
    from mcp.neural_network_models.diffusion_model import DiffusionModel
    
    print("Importing genetic_diffusion_model...")
    from mcp.neural_network_models.genetic_diffusion_model import GeneticDiffusionModel
    
    print("Importing brain_state_integration...")
    from mcp.neural_network_models.brain_state_integration import BrainStateIntegration
    
    print("All imports successful!")
    sys.exit(0)
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)